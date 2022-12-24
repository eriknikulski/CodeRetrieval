from __future__ import unicode_literals, print_function, division
import argparse
import pickle
import random
import string
import time
from contextlib import contextmanager
from enum import Enum

import comet_ml
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


def get_grad_norm(model):
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) > 0:
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2).to(device, non_blocking=True) for p in parameters]), 2.0).item()
    return total_norm


def get_correct(results, targets):
    targets_mask = targets != const.PAD_TOKEN
    device = targets.device
    results_masked = results.where(targets_mask, torch.tensor(-1, device=device))
    targets_masked = targets.where(targets_mask, torch.tensor(-1, device=device))
    return (results_masked == targets_masked).all(axis=1).sum()


def get_accuracies(outputs_seqs, doc_seqs, code_seqs, arch, batch_size):
    if arch.n_decoders == 2:
        return torch.stack((get_correct(outputs_seqs[0], doc_seqs) / batch_size,
                            get_correct(outputs_seqs[1], code_seqs) / batch_size))

    if arch.Type.DOC in arch.decoders:
        return (get_correct(outputs_seqs[0], doc_seqs) / batch_size).unsqueeze(0)
    if arch.Type.CODE in arch.decoders:
        return (get_correct(outputs_seqs[0], code_seqs) / batch_size).unsqueeze(0)


def get_decoder_loss(loss_fn, decoder_outputs, targets):
    loss = 0
    target_length = targets.shape[0]

    for i in range(target_length):
        current_loss = loss_fn(decoder_outputs[i], targets[i])
        loss += current_loss
    return loss / target_length


def criterion(loss_fn, outputs, targets):
    loss = 0
    for i, output in enumerate(outputs):
        loss += get_decoder_loss(loss_fn, output.permute(1, 0, 2), targets[i].T)
    return loss


def go(mode: Mode, arch: model.Architecture, joint_embedder, optimizer, dataloader, loss_fn, scaler, config,
       experiment=None, epoch=0):
    joint_module = getattr(joint_embedder, 'module', joint_embedder)      # get module if wrapped in DDP
    doc_lang = dataloader.dataset.doc_lang
    code_lang = dataloader.dataset.code_lang

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    size = len(dataloader.dataset) / world_size
    num_batches = int(size / config['batch_size'])
    
    epoch_loss = torch.zeros(1, device=config['device'])
    epoch_accuracies = torch.zeros(arch.n_decoders, device=config['device'])
    outputs_seqs = None
    encoder_id = None
    encoder_inputs = None
    
    if mode == Mode.TRAIN:
        joint_embedder.train()
    else:
        joint_embedder.eval()
        torch.set_grad_enabled(False)
    
    for batch, batch_data in enumerate(dataloader):
        doc_seqs, doc_seq_lengths, code_seqs, code_seq_lengths, methode_names, methode_name_lengths, code_tokens = \
            (elem.to(config['device'], non_blocking=True) if torch.is_tensor(elem) else elem for elem in batch_data)
        
        if mode == Mode.TRAIN:
            optimizer.zero_grad(set_to_none=const.SET_GRADIENTS_NONE)

        encoder_id = arch.get_rand_encoder_id()
        encoder_inputs = arch.get_encoder_input(encoder_id, doc_seqs, doc_seq_lengths, code_seqs, code_seq_lengths,
                                                methode_names, methode_name_lengths, code_tokens)
        decoder_sizes = arch.get_decoder_sizes(doc_seqs[0].size(0), code_seqs[0].size(0))

        with optional(config['fp16'] and scaler, torch.cuda.amp.autocast):
            decoders_outputs, outputs_seqs = joint_embedder(encoder_id, encoder_inputs, decoder_sizes)
        batch_loss = criterion(loss_fn, decoders_outputs, [doc_seqs, code_seqs])

        if mode == Mode.TRAIN:
            if scaler:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()

        if const.LOG_IN_TRAINING or mode != Mode.TRAIN:
            epoch_loss += batch_loss
            batch_accuracies = get_accuracies(outputs_seqs, doc_seqs, code_seqs, arch, config['batch_size'])
            epoch_accuracies += batch_accuracies

            if experiment and mode == Mode.TRAIN and const.LOG_BATCHES:
                grad_norms = [get_grad_norm(module) for module_list in joint_module.children()
                              for module in module_list.children()]
                experiment.log_batch_metrics(mode.value, batch_loss, batch_accuracies, grad_norms,
                                             step=epoch * size / config['batch_size'] + batch)

    if const.LOG_IN_TRAINING or mode != Mode.TRAIN:
        epoch_loss /= num_batches
        epoch_accuracies /= num_batches

        if experiment:
            outputs_seqs = [out[:5] for out in outputs_seqs]
            input_lang = arch.get_encoder_lang(encoder_id, doc_lang, code_lang)
            output_langs = arch.get_decoder_languages(doc_lang, code_lang)
            translations = comet.generate_text_seq(input_lang, output_langs, encoder_inputs[0][:5], outputs_seqs, epoch)
            experiment.log_epoch_metrics(mode.value, epoch_loss, epoch_accuracies, translations, epoch=epoch)

    torch.set_grad_enabled(True)
    return epoch_loss, torch.mean(epoch_accuracies)


def train_loop(joint_embedder, arch, optimizer, dataloader, loss_fn, scaler, config, experiment=None, epoch=0):
    return go(Mode.TRAIN, arch, joint_embedder, optimizer, dataloader, loss_fn, scaler, config, experiment, epoch)


def valid_loop(joint_embedder, arch, dataloader, loss_fn, config, experiment=None, epoch=0):
    return go(Mode.VALID, arch, joint_embedder, None, dataloader, loss_fn, None, config, experiment, epoch)


def test_loop(joint_embedder, arch, dataloader, loss_fn, config, experiment=None, epoch=0):
    return go(Mode.TEST, arch, joint_embedder, None, dataloader, loss_fn, None, config, experiment, epoch)


def go_train(rank, world_size, arch_mode, experiment_name, port, train_data=None, valid_data=None):
    if rank is not None:
        ddp.setup(rank, world_size, port)

    if not train_data:
        with open(const.DATA_WORKING_TRAIN_PATH, 'rb') as train_file:
            train_data = pickle.load(train_file)
    if not valid_data:
        with open(const.DATA_WORKING_VALID_PATH, 'rb') as valid_file:
            valid_data = pickle.load(valid_file)

    train_sampler = None
    valid_sampler = None

    scaler = None

    arch = model.Architecture(arch_mode)
    joint_embedder = model.JointEmbedder(arch, train_data.input_lang.n_words, train_data.output_lang.n_words)

    experiment = Experiment(experiment_name)
    experiment.log_initial_params(world_size, arch, len(train_data), len(valid_data),
                                  train_data.input_lang.n_words, train_data.output_lang.n_words)

    if ddp.is_dist_avail_and_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=const.SHUFFLE_DATA)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, shuffle=const.SHUFFLE_DATA)

        joint_embedder = DistributedDataParallel(joint_embedder.to(const.DEVICE),
                                                 find_unused_parameters=const.DDP_FIND_UNUSED_PARAMETER,
                                                 device_ids=[rank])
        scaler = torch.cuda.amp.GradScaler()

    shuffle = const.SHUFFLE_DATA if (train_sampler is None) else None
    dataloader = loader.DataLoader(train_data, batch_size=const.BATCH_SIZE, shuffle=shuffle,
                                   collate_fn=pad_collate.collate, sampler=train_sampler, drop_last=True,
                                   num_workers=const.NUM_WORKERS_DATALOADER, pin_memory=const.PIN_MEMORY)
    valid_dataloader = loader.DataLoader(valid_data, batch_size=const.BATCH_SIZE, shuffle=shuffle,
                                         collate_fn=pad_collate.collate, sampler=valid_sampler, drop_last=True,
                                         num_workers=const.NUM_WORKERS_DATALOADER, pin_memory=const.PIN_MEMORY)

    loss_fn = nn.NLLLoss(ignore_index=const.PAD_TOKEN if const.IGNORE_PADDING_IN_LOSS else -100)
    optimizer = optim.SGD(joint_embedder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=const.LR_STEP_SIZE, gamma=const.LR_GAMMA)
    
    config = {
        'batch_size': const.BATCH_SIZE,
        'bidirectional': const.BIDIRECTIONAL,
        'encoder_layers': const.ENCODER_LAYERS,
        'hidden_size': const.HIDDEN_SIZE,
        'grad_clipping_enabled': const.GRADIENT_CLIPPING_ENABLED,
        'grad_clipping_max_norm': const.GRADIENT_CLIPPING_MAX_NORM,
        'grad_clipping_norm_type': const.GRADIENT_CLIPPING_NORM_TYPE,
        'device': const.DEVICE,
        'fp16': const.FP16,
    }

    with profiler.Profiler(active=const.PROFILER_IS_ACTIVE) as p:
        for epoch in range(const.EPOCHS):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            if train_sampler and valid_sampler:
                train_sampler.set_epoch(epoch)
                valid_sampler.set_epoch(epoch)

            train_loop(joint_embedder, arch, optimizer, dataloader, loss_fn, scaler, config, experiment, epoch=epoch)
            valid_loss, valid_accuracy = valid_loop(joint_embedder, arch, valid_dataloader, loss_fn, config, experiment,
                                                    epoch=epoch)

            experiment.log_learning_rate(optimizer.param_groups[0]['lr'], epoch=epoch)
            save.checkpoint_encoders_decoders(epoch, joint_embedder, valid_loss,
                                              const.CHECKPOINT_PATH + const.SLURM_JOB_ID)

            scheduler.step()
            p.step()

    save.model(joint_embedder.state_dict(), const.MODEL_JOINT_EMBEDDER_PATH)
    if ddp.is_dist_avail_and_initialized():
        ddp.cleanup()
    experiment.end()


@print_time('\nTotal ')
def run(args):
    if args.data == 'java':
        data_path = const.JAVA_PATH
    else:
        data_path = const.SYNTH_PATH

    remove_duplicates = not args.keep_duplicates
    arch_mode = model.Architecture.Mode(args.architecture)

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

        if not const.SHUFFLE_DATA:
            train_data.sort()
            test_data.sort()
            valid_data.sort()
    elif not args.last_data:
        input_lang = data.Lang('doc')
        output_lang = data.Lang('code')
        train_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'train/', languages=[input_lang, output_lang],
                                        remove_duplicates=remove_duplicates)
        test_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'test/', languages=[input_lang, output_lang],
                                       remove_duplicates=remove_duplicates)
        valid_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'valid/', languages=[input_lang, output_lang],
                                        remove_duplicates=remove_duplicates)

    if not args.last_data:
        pickle.dump(train_data, open(const.DATA_WORKING_TRAIN_PATH, 'wb'))
        pickle.dump(test_data, open(const.DATA_WORKING_TEST_PATH, 'wb'))
        pickle.dump(valid_data, open(const.DATA_WORKING_VALID_PATH, 'wb'))

    experiment_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(const.COMET_EXP_NAME_LENGTH))
    port = ddp.find_free_port(const.MASTER_ADDR)

    print(f'CUDA_DEVICE_COUNT: {const.CUDA_DEVICE_COUNT}')
    if args.gpu:
        ddp.run(go_train, const.CUDA_DEVICE_COUNT, arch_mode, experiment_name, port)
    else:
        if not args.last_data:
            go_train(None, 1, arch_mode, experiment_name, port, train_data, valid_data)
        else:
            go_train(None, 1, arch_mode, experiment_name, port)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
