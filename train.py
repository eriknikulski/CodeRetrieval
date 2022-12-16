from __future__ import unicode_literals, print_function, division
import argparse
import pickle
import random
import statistics
import string
import time
from enum import Enum
from operator import add

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
parser.add_argument('-lo', '--labels-only', action='store_true', default=False, help='The data to be used.')
parser.add_argument('-ld', '--load-data', action='store_true', default=False, help='Load preprocessed data.')
parser.add_argument('-kd', '--keep-duplicates', action='store_true', default=False, help='Do not remove duplicates in data.')
parser.add_argument('-g', '--gpu', action='store_true', default=False, help='Run on GPU(s).')
parser.add_argument('-lad', '--last-data', action='store_true', default=False, help='Use last working dataset.')


class Mode(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


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
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item()
    return total_norm


def get_correct(results, targets):
    targets_mask = targets != const.PAD_TOKEN
    device = targets.device
    results_masked = results.where(targets_mask, torch.tensor(-1, device=device))
    targets_masked = targets.where(targets_mask, torch.tensor(-1, device=device))
    return (results_masked == targets_masked).all(axis=1).sum().item()


def get_decoder_loss(loss_fn, decoder_outputs, targets, ignore_padding=const.IGNORE_PADDING_IN_LOSS):
    loss = 0
    device = targets.device
    target_length = targets.shape[0]

    for i in range(target_length):
        current_target = targets[i]
        decoder_output = decoder_outputs[i]
        current_loss = loss_fn(decoder_output, current_target)
    
        if ignore_padding:
            loss_mask = current_target != const.PAD_TOKEN
            loss_masked = current_loss.where(loss_mask, torch.tensor(0.0, device=device))
            current_loss = loss_masked.sum() / loss_mask.sum() if loss_mask.sum() else 0
        loss += current_loss
    return loss / target_length


def go(mode: Mode, joint_embedder, optimizers, dataloader, loss_fn, config, experiment=None, epoch=0):
    joint_module = getattr(joint_embedder, 'module', joint_embedder)      # get module if wrapped in DDP
    n_decoders = len(joint_module.decoders)
    input_lang = dataloader.dataset.input_lang
    output_lang = dataloader.dataset.output_lang

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    size = len(dataloader.dataset) / world_size
    num_batches = int(size / config['batch_size'])
    
    epoch_loss = 0
    epoch_accuracies = []
    inputs = None
    outputs_seqs = None
    
    if mode == Mode.TRAIN:
        joint_embedder.train()
    else:
        joint_embedder.eval()
        torch.set_grad_enabled(False)
    
    for batch, (inputs, targets, urls) in enumerate(dataloader):
        inputs, targets = inputs.to(config['device']), targets.to(config['device'])

        batch_loss = 0
        batch_accuracies = []
        
        if mode == Mode.TRAIN:
            for optimizer in optimizers:
                optimizer.zero_grad()

        input_length = inputs[0].size(0)
        target_length = targets[0].size(0)

        decoders_outputs, outputs_seqs = joint_embedder(inputs, target_length)
        
        for decoder_outputs in decoders_outputs:
            batch_loss += get_decoder_loss(loss_fn, decoder_outputs.permute(1, 0, 2), targets.T, 
                                           config['ignore_padding_in_loss'])
        if mode == Mode.TRAIN:
            batch_loss.backward()
            for optimizer in optimizers:
                optimizer.step()

        batch_loss = batch_loss.item()
        epoch_loss += batch_loss
        # calc percentage of correctly generated sequences
        batch_accuracies.append(get_correct(outputs_seqs[-1], targets) / config['batch_size'])          # acc regen code
        if n_decoders > 1:
            batch_accuracies.append(get_correct(outputs_seqs[0], inputs) / config['batch_size'])        # acc regen doc
        epoch_accuracies = map(add, epoch_accuracies, batch_accuracies) if epoch_accuracies else batch_accuracies

        if experiment and mode == Mode.TRAIN and const.LOG_BATCHES:
            grad_norms = [get_grad_norm(module) for module_list in joint_module.children() 
                          for module in module_list.children()]
            experiment.log_batch_metrics(mode.value, batch_loss, batch_accuracies, grad_norms,
                                         step=epoch * size / config['batch_size'] + batch)
            
    epoch_loss /= num_batches
    epoch_accuracies = list(map(lambda x: x / num_batches, epoch_accuracies))
    
    if experiment:
        translations = comet.generate_text_seq(input_lang, output_lang, inputs[:5], outputs_seqs[::, :5], epoch)
        experiment.log_epoch_metrics(mode.value, epoch_loss, epoch_accuracies, translations, epoch=epoch)

    torch.set_grad_enabled(True)
    return epoch_loss, statistics.mean(epoch_accuracies)


def train_loop(joint_embedder, optimizers, dataloader, loss_fn, config, experiment=None, epoch=0):
    return go(Mode.TRAIN, joint_embedder, optimizers, dataloader, loss_fn, config, experiment, epoch)


def valid_loop(joint_embedder, dataloader, loss_fn, config, experiment=None, epoch=0):
    return go(Mode.VALID, joint_embedder, None, dataloader, loss_fn, config, experiment, epoch)


def test_loop(joint_embedder, dataloader, loss_fn, config, experiment=None, epoch=0):
    return go(Mode.TEST, joint_embedder, None, dataloader, loss_fn, config, experiment, epoch)


def go_train(rank, world_size, experiment_name, port, train_data=None, valid_data=None):
    if rank is not None:
        ddp.setup(rank, world_size, port)

    if not train_data:
        with open(const.DATA_WORKING_TRAIN_PATH, 'rb') as train_file:
            train_data = pickle.load(train_file)
    if not valid_data:
        with open(const.DATA_WORKING_VALID_PATH, 'rb') as valid_file:
            valid_data = pickle.load(valid_file)

    input_lang = train_data.input_lang
    output_lang = train_data.output_lang

    train_sampler = None
    valid_sampler = None

    experiment = Experiment(experiment_name)
    experiment.log_initial_metrics(world_size, len(train_data), len(valid_data), input_lang.n_words, output_lang.n_words)

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE, const.BATCH_SIZE, input_lang)
    decoder = model.DecoderRNNWrapped(const.HIDDEN_SIZE, output_lang.n_words, const.BATCH_SIZE, output_lang)
    joint_embedder = model.JointEmbeder([encoder], [decoder])

    if ddp.is_dist_avail_and_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=const.SHUFFLE_DATA)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, shuffle=const.SHUFFLE_DATA)

        joint_embedder = DistributedDataParallel(joint_embedder.to(const.DEVICE), device_ids=[rank])

    shuffle = const.SHUFFLE_DATA if (train_sampler is None) else None
    dataloader = loader.DataLoader(train_data, batch_size=const.BATCH_SIZE, shuffle=shuffle,
                                   collate_fn=pad_collate.collate, sampler=train_sampler, drop_last=True,
                                   num_workers=const.NUM_WORKERS_DATALOADER, pin_memory=const.PIN_MEMORY)
    valid_dataloader = loader.DataLoader(valid_data, batch_size=const.BATCH_SIZE, shuffle=shuffle,
                                         collate_fn=pad_collate.collate, sampler=valid_sampler, drop_last=True,
                                         num_workers=const.NUM_WORKERS_DATALOADER, pin_memory=const.PIN_MEMORY)

    loss_fn = nn.NLLLoss(reduction='none') if const.IGNORE_PADDING_IN_LOSS else nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)
    optimizers = [encoder_optimizer, decoder_optimizer]

    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=const.LR_STEP_SIZE, gamma=const.LR_GAMMA)
    decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=const.LR_STEP_SIZE, gamma=const.LR_GAMMA)
    
    config = {
        'batch_size': const.BATCH_SIZE,
        'bidirectional': const.BIDIRECTIONAL,
        'encoder_layers': const.ENCODER_LAYERS,
        'hidden_size': const.HIDDEN_SIZE,
        'ignore_padding_in_loss': const.IGNORE_PADDING_IN_LOSS,
        'grad_clipping_enabled': const.GRADIENT_CLIPPING_ENABLED,
        'grad_clipping_max_norm': const.GRADIENT_CLIPPING_MAX_NORM,
        'grad_clipping_norm_type': const.GRADIENT_CLIPPING_NORM_TYPE,
        'device': const.DEVICE,
    }

    with profiler.Profiler(active=const.PROFILER_IS_ACTIVE) as p:
        for epoch in range(const.EPOCHS):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            if train_sampler and valid_sampler:
                train_sampler.set_epoch(epoch)
                valid_sampler.set_epoch(epoch)

            train_loop(joint_embedder, optimizers, dataloader, loss_fn, config, experiment, epoch=epoch)
            valid_loss, valid_accuracy = valid_loop(joint_embedder, valid_dataloader, loss_fn, config, experiment, 
                                                    epoch=epoch)

            experiment.log_learning_rate(encoder_optimizer.param_groups[0]['lr'],
                                         decoder_optimizer.param_groups[0]['lr'], epoch=epoch)
            save.checkpoint_encoders_decoders(epoch, joint_embedder, optimizers, valid_loss,
                                              const.CHECKPOINT_PATH + const.SLURM_JOB_ID)

            encoder_scheduler.step()
            decoder_scheduler.step()
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

    if args.labels_only:
        const.LABELS_ONLY = True

    if args.keep_duplicates:
        remove_duplicates = False
    else:
        remove_duplicates = True

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

        if const.LABELS_ONLY:
            train_data.df['code_tokens'] = train_data.df['docstring_tokens']
            test_data.df['code_tokens'] = test_data.df['docstring_tokens']
            valid_data.df['code_tokens'] = valid_data.df['docstring_tokens']

            train_data.output_lang = train_data.input_lang
            test_data.output_lang = test_data.input_lang
            valid_data.output_lang = valid_data.input_lang

            if not const.SHUFFLE_DATA:
                train_data.sort()
                test_data.sort()
                valid_data.sort()

            train_data.to_numpy()
            test_data.to_numpy()
            valid_data.to_numpy()
    elif not args.last_data:
        input_lang = data.Lang('docstring')
        output_lang = data.Lang('code')
        train_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'train/',
                                        labels_only=const.LABELS_ONLY, languages=[input_lang, output_lang],
                                        remove_duplicates=remove_duplicates)
        test_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'test/',
                                       labels_only=const.LABELS_ONLY, languages=[input_lang, output_lang],
                                       remove_duplicates=remove_duplicates)
        valid_data = loader.CodeDataset(const.PROJECT_PATH + data_path + 'valid/',
                                        labels_only=const.LABELS_ONLY, languages=[input_lang, output_lang],
                                        remove_duplicates=remove_duplicates)

    if not args.last_data:
        pickle.dump(train_data, open(const.DATA_WORKING_TRAIN_PATH, 'wb'))
        pickle.dump(test_data, open(const.DATA_WORKING_TEST_PATH, 'wb'))
        pickle.dump(valid_data, open(const.DATA_WORKING_VALID_PATH, 'wb'))

    experiment_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(const.COMET_EXP_NAME_LENGTH))
    port = ddp.find_free_port(const.MASTER_ADDR)

    print(f'CUDA_DEVICE_COUNT: {const.CUDA_DEVICE_COUNT}')
    if args.gpu:
        ddp.run(go_train, const.CUDA_DEVICE_COUNT, experiment_name, port)
    else:
        if not args.last_data:
            go_train(None, 1, experiment_name, port, train_data, valid_data)
        else:
            go_train(None, 1, experiment_name, port)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
