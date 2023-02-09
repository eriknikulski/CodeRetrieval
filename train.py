from __future__ import unicode_literals, print_function, division
import argparse
import pickle
import random
import string
import time
from contextlib import contextmanager
from enum import Enum

import comet_ml
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel

import comet
import save
from comet import Experiment
import const
import data
import ddp
import loader
import model
import pad_collate
import profiler


parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-d', '--data', choices=['java', 'synth'], help='The data to be used.')
parser.add_argument('-ld', '--load-data', action='store_true', default=False, help='Load preprocessed data.')
parser.add_argument('-kd', '--keep-duplicates', action='store_true', default=False,
                    help='Do not remove duplicates in data.')
parser.add_argument('-g', '--gpu', action='store_true', default=False, help='Run on GPU(s).')
parser.add_argument('-lad', '--last-data', action='store_true', default=False, help='Use last working dataset.')
parser.add_argument('-arch', '--architecture', choices=['normal', 'doc_doc', 'code_code', 'doc_code', 'code_doc'],
                    default='normal', help='The model architecture to be used for training.')
parser.add_argument('-m', '--model', choices=['translator', 'embedder'], help='What model to use.')
parser.add_argument('-sce', '--simple-code-encoder', action='store_true', default=False, help='Use simple Code Encoder')


class Mode(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


@contextmanager
def optional(condition, context_manager):
    if condition:
        with context_manager():
            yield
    else:
        yield


def print_time(prefix=''):
    def wrapper(func):
        def inner(*args, **kwargs):
            begin = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            print(prefix + 'time taken: ', int(end - begin), 's')
            return res
        return inner
    return wrapper


def seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


class Trainer:
    def __init__(self, arch: model.Architecture, module: nn.Module,
                 data: tuple[loader.CodeDataset, loader.CodeDataset, loader.CodeDataset], config,
                 experiment: comet.Experiment):
        self.arch = arch
        self.model = module
        self.config = config
        self.experiment = experiment
        self.device = self.config['device']         # or just use config??

        self.train_data, self.valid_data, self.test_data = data

        self.doc_lang = self.train_data.lang
        self.code_lang = self.train_data.lang

        self.model_type = model.ModelType.TRANSLATOR if isinstance(self.model, model.JointTranslator) \
            else model.JointEmbedder

        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None
        self.scaler = None

        if ddp.is_dist_avail_and_initialized():
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data,
                                                                                 shuffle=self.config['shuffle_data'])
            self.valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_data,
                                                                                 shuffle=self.config['shuffle_data'])
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_data,
                                                                                shuffle=self.config['shuffle_data'])

            self.model = DistributedDataParallel(self.model.to(self.device),
                                                 find_unused_parameters=self.config['ddp_find_unused_parameter'],
                                                 device_ids=[self.config['rank']])
            self.scaler = torch.cuda.amp.GradScaler()

        shuffle = self.config['shuffle_data'] if (self.train_sampler is None) else None
        self.train_dataloader = loader.DataLoader(self.train_data, batch_size=self.config['batch_size'],
                                                  shuffle=shuffle, collate_fn=pad_collate.collate,
                                                  sampler=self.train_sampler, drop_last=True,
                                                  num_workers=self.config['num_workers_dataloader'],
                                                  pin_memory=self.config['pin_memory'])
        self.valid_dataloader = loader.DataLoader(self.valid_data, batch_size=self.config['batch_size'],
                                                  shuffle=shuffle, collate_fn=pad_collate.collate,
                                                  sampler=self.valid_sampler, drop_last=True,
                                                  num_workers=self.config['num_workers_dataloader'],
                                                  pin_memory=self.config['pin_memory'])
        self.test_dataloader = loader.DataLoader(self.test_data, batch_size=self.config['batch_size'],
                                                 shuffle=shuffle, collate_fn=pad_collate.collate,
                                                 sampler=self.test_sampler, drop_last=True,
                                                 num_workers=self.config['num_workers_dataloader'],
                                                 pin_memory=self.config['pin_memory'])

        self.loss_fn = nn.NLLLoss(ignore_index=self.config['pad_token']
                                  if self.config['ignore_padding_in_loss'] else -100)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                   momentum=self.config['momentum'])

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['lr_step_size'],
                                                   gamma=self.config['lr_gamma'])

    def batch_data_to_device(self, batch):
        return tuple(elem.to(self.device, non_blocking=True) if torch.is_tensor(elem) else elem for elem in batch)

    def get_correct(self, results, targets):
        targets_mask = targets != const.PAD_TOKEN
        results_masked = results.where(targets_mask, torch.tensor(-1, device=self.device))
        targets_masked = targets.where(targets_mask, torch.tensor(-1, device=self.device))
        return (results_masked == targets_masked).all(axis=1).sum()

    def get_accuracies(self, outputs_seqs, doc_seqs, code_seqs):
        if self.arch.n_decoders == 2:
            return torch.stack((self.get_correct(outputs_seqs[0], doc_seqs) / self.config['batch_size'],
                                self.get_correct(outputs_seqs[1], code_seqs) / self.config['batch_size']))

        if self.arch.Type.DOC in self.arch.decoders:
            return (self.get_correct(outputs_seqs[0], doc_seqs) / self.config['batch_size']).unsqueeze(0)
        if self.arch.Type.CODE in self.arch.decoders:
            return (self.get_correct(outputs_seqs[0], code_seqs) / self.config['batch_size']).unsqueeze(0)

    def get_decoder_loss(self, decoder_outputs, targets):
        loss = 0
        target_length = targets.shape[0]

        for i in range(target_length):
            current_loss = self.loss_fn(decoder_outputs[i], targets[i])
            loss += current_loss
        return loss / target_length

    def criterion(self, outputs, targets):
        loss = 0
        for i, output in enumerate(outputs):
            loss += self.get_decoder_loss(output.permute(1, 0, 2), targets[i].T)
        return loss

    def _go(self, mode: Mode, dataloader: loader.DataLoader, epoch: int):
        data_size = len(dataloader.dataset) / self.config['world_size']
        num_batches = int(data_size / self.config['batch_size'])

        epoch_loss = torch.zeros(1, device=self.config['device'])
        epoch_accuracies = torch.zeros(self.arch.n_decoders, device=self.config['device'])
        batch_accuracies = torch.zeros(self.arch.n_decoders, device=self.config['device'])
        batch_accuracies_list = []
        outputs_seqs = None
        encoder_id = None
        encoder_inputs = None

        translations = None

        if mode == Mode.TRAIN:
            self.model.train()
        else:
            self.model.eval()

        with optional(mode != Mode.TRAIN, torch.no_grad):
            for batch, batch_data in enumerate(dataloader):
                batch_data = self.batch_data_to_device(batch_data)

                doc_inputs = batch_data[:2]
                code_inputs = batch_data[2:8]
                neg_doc_inputs = batch_data[8:10]
                neg_code_inputs = batch_data[10:]

                doc_seqs, doc_seq_lengths = doc_inputs
                code_seqs, code_seq_lengths, methode_names, methode_name_lengths, code_tokens, code_tokens_length = code_inputs

                if mode == Mode.TRAIN:
                    self.optimizer.zero_grad(set_to_none=self.config['set_gradients_none'])

                if self.model_type == model.ModelType.TRANSLATOR:
                    encoder_id = self.arch.get_rand_encoder_id()
                    encoder_inputs = self.arch.get_encoder_input(encoder_id, doc_inputs, code_inputs)
                    decoder_sizes = self.arch.get_decoder_sizes(doc_seqs[0].size(0), code_seqs[0].size(0))
                    targets = self.arch.get_decoder_targets(doc_seqs, code_seqs)

                    with optional(self.config['fp16'] and self.scaler, torch.cuda.amp.autocast):
                        decoders_outputs, outputs_seqs = self.model(encoder_id, encoder_inputs, decoder_sizes)
                    batch_loss = self.criterion(decoders_outputs, targets)
                else:
                    with optional(self.config['fp16'] and self.scaler, torch.cuda.amp.autocast):
                        batch_loss = self.model(doc_inputs, code_inputs, neg_doc_inputs, neg_code_inputs)

                # TODO: torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) after backward() before step() ?? necessary??
                if mode == Mode.TRAIN:
                    if self.scaler:
                        self.scaler.scale(batch_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        batch_loss.backward()
                        self.optimizer.step()

                if self.config['log_in_training'] or mode != Mode.TRAIN:
                    epoch_loss += batch_loss
                    if self.model_type == model.ModelType.TRANSLATOR:
                        batch_accuracies = self.get_accuracies(outputs_seqs, doc_seqs, code_seqs)
                        epoch_accuracies += batch_accuracies
                        batch_accuracies_list.append(batch_accuracies)

                    if self.experiment and mode == Mode.TRAIN:
                        # TODO: rework gradients
                        grad_norms = [get_grad_norm(module) for module_list in self.model.children()
                                      for module in module_list.children()]
                        self.experiment.log_metrics(mode.value, batch_loss, batch_accuracies, grad_norms,
                                                    step=epoch * data_size / self.config['batch_size'] + batch)

            if self.config['log_in_training'] or mode != Mode.TRAIN:
                epoch_loss /= num_batches
                epoch_accuracies /= num_batches

                if self.experiment:
                    if self.model_type == model.ModelType.TRANSLATOR:
                        outputs_seqs = [out[:5] for out in outputs_seqs]
                        input_lang = self.arch.get_encoder_lang(encoder_id, self.doc_lang, self.code_lang)
                        output_langs = self.arch.get_decoder_languages(self.doc_lang, self.code_lang)
                        translations = comet.generate_text_seq(input_lang, output_langs, encoder_inputs[0][:5],
                                                               outputs_seqs, epoch)
                    self.experiment.log_metrics(mode.value, epoch_loss, epoch_accuracies, text=translations,
                                                epoch=epoch)
                    self.experiment.log_acc_std_mean(mode.value, batch_accuracies_list, epoch=epoch)

        return epoch_loss, torch.mean(epoch_accuracies)

    def _train(self, epoch: int):
        return self._go(Mode.TRAIN, self.train_dataloader, epoch)

    def _valid(self, epoch: int):
        return self._go(Mode.VALID, self.valid_dataloader, epoch)

    def _test(self):
        return self._go(Mode.TEST, self.test_dataloader, -1)

    def save_checkpoint(self, epoch, loss):
        save.checkpoint_encoders_decoders(epoch, self.model, loss,
                                          self.config['checkpoint_path'] + self.config['slurm_job_id'])

    def save_model(self):
        save.model(self.model.state_dict(), self.config['model_save_path'])

    def cleanup(self):
        if ddp.is_dist_avail_and_initialized():
            ddp.cleanup()
        self.experiment.end()

    def train(self):
        with profiler.Profiler(active=const.PROFILER_IS_ACTIVE) as p:
            for epoch in range(self.config['epochs']):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                if self.train_sampler and self.valid_sampler:
                    self.train_sampler.set_epoch(epoch)
                    self.valid_sampler.set_epoch(epoch)

                self._train(epoch)
                valid_loss, valid_accuracy = self._valid(epoch)

                self.experiment.log_learning_rate(self.optimizer.param_groups[0]['lr'], epoch=epoch)
                self.save_checkpoint(epoch, valid_loss)

                self.scheduler.step()
                p.step()
            test_loss, test_accuracy = self._test()
            self.experiment.log_metrics(Mode.TEST, test_loss, test_accuracy, epoch=epoch)

        self.save_model()
        self.cleanup()


def get_grad_norm(model):
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) > 0:
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2).to(device, non_blocking=True) for p in parameters]), 2.0).item()
    return total_norm


def go_train(rank, world_size, arch_mode, module, experiment_name, port, train_data: loader.CodeDataset = None,
             valid_data: loader.CodeDataset = None, test_data: loader.CodeDataset = None):
    if rank is not None:
        ddp.setup(rank, world_size, port)

    if not train_data:
        with open(const.DATA_WORKING_TRAIN_PATH, 'rb') as train_file:
            train_data = pickle.load(train_file)
    if not valid_data:
        with open(const.DATA_WORKING_VALID_PATH, 'rb') as valid_file:
            valid_data = pickle.load(valid_file)
    if not test_data:
        with open(const.DATA_WORKING_TEST_PATH, 'rb') as test_file:
            test_data = pickle.load(test_file)

    config = {
        'rank': rank,
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
        'slurm_job_id': const.SLURM_JOB_ID,

        'model_save_path': const.MODEL_JOINT_TRANSLATOR_PATH,  # TODO: generalize
        'checkpoint_path': const.CHECKPOINT_PATH,

        'pad_token': const.PAD_TOKEN,

        'ignore_padding_in_loss': const.IGNORE_PADDING_IN_LOSS,
        'shuffle_data': const.SHUFFLE_DATA,
        'num_workers_dataloader': const.NUM_WORKERS_DATALOADER,
        'pin_memory': const.PIN_MEMORY,
        'log_in_training': const.LOG_IN_TRAINING,
        'fp16': const.FP16,
        'set_gradients_none': const.SET_GRADIENTS_NONE,
        'ddp_find_unused_parameter': const.DDP_FIND_UNUSED_PARAMETER,

        'bidirectional': const.BIDIRECTIONAL,
        'encoder_layers': const.ENCODER_LAYERS,
        'hidden_size': const.HIDDEN_SIZE,

        'learning_rate': const.LEARNING_RATE,
        'momentum': const.MOMENTUM,
        'epochs': const.EPOCHS,
        'batch_size': const.BATCH_SIZE,
        'lr_step_size': const.LR_STEP_SIZE,
        'lr_gamma': const.LR_GAMMA,

        'device': const.DEVICE,
    }

    arch = model.Architecture(arch_mode)

    experiment = Experiment(experiment_name)
    experiment.log_initial_params(world_size, arch_mode, len(train_data), len(valid_data),
                                  train_data.lang.n_words, train_data.lang.n_words)

    trainer = Trainer(arch, module, (train_data, valid_data, test_data), config, experiment)
    trainer.train()


@print_time('\nTotal ')
def run(args):
    if args.data == 'java':
        data_path = const.JAVA_PATH
    else:
        data_path = const.SYNTH_PATH

    remove_duplicates = not args.keep_duplicates
    arch_mode = model.Architecture.Mode(args.architecture)
    const.SIMPLE_CODE_ENCODER = args.simple_code_encoder

    const.CUDA_DEVICE_COUNT = torch.cuda.device_count()
    if args.gpu and const.CUDA_DEVICE_COUNT < 1:
        raise Exception('When running in GPU mode there should be at least 1 GPU available')
    if args.load_data and args.last_data:
        raise Exception('Unclear data argument. Choose one!')

    if args.load_data:
        with open(const.DATA_TRAIN_PATH, 'rb') as train_file:
            train_data = pickle.load(train_file)
            train_data.enforce_length_constraints()
        with open(const.DATA_TEST_PATH, 'rb') as test_file:
            test_data = pickle.load(test_file)
            test_data.enforce_length_constraints()
        with open(const.DATA_VALID_PATH, 'rb') as valid_file:
            valid_data = pickle.load(valid_file)
            valid_data.enforce_length_constraints()
    elif not args.last_data:
        lang = data.Lang('lang')
        train_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'train/', language=lang,
                                        remove_duplicates=remove_duplicates)
        test_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'test/', language=lang,
                                       remove_duplicates=remove_duplicates)
        valid_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'valid/', language=lang,
                                        remove_duplicates=remove_duplicates)

    if not args.last_data:
        pickle.dump(train_data, open(const.DATA_WORKING_TRAIN_PATH, 'wb'))
        pickle.dump(test_data, open(const.DATA_WORKING_TEST_PATH, 'wb'))
        pickle.dump(valid_data, open(const.DATA_WORKING_VALID_PATH, 'wb'))

    if args.model == 'translator':
        arch = model.Architecture(arch_mode)
        module = model.JointTranslator(arch, train_data.lang.n_words, train_data.lang.n_words,
                                       simple=const.SIMPLE_CODE_ENCODER)
    else:
        module = model.JointEmbedder(train_data.lang.n_words, train_data.lang.n_words, simple=const.SIMPLE_CODE_ENCODER)
        const.LEARNING_RATE = 1.34e-4

    experiment_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(const.COMET_EXP_NAME_LENGTH))
    port = ddp.find_free_port(const.MASTER_ADDR)

    print(f'CUDA_DEVICE_COUNT: {const.CUDA_DEVICE_COUNT}')
    if args.gpu:
        ddp.run(go_train, const.CUDA_DEVICE_COUNT, arch_mode, module, experiment_name, port)
    else:
        if not args.last_data:
            go_train(None, 1, arch_mode, module, experiment_name, port, train_data, valid_data)
        else:
            go_train(None, 1, arch_mode, module, experiment_name, port)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
