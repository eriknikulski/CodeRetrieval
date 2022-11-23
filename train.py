from __future__ import unicode_literals, print_function, division
import argparse
import pickle
import random
import string
import time

import comet_ml
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from comet import Experiment
import const
import data
import ddp
import loader
import model
import pad_collate


parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-d', '--data', choices=['java', 'synth'], help='The data to be used.')
parser.add_argument('-lo', '--labels-only', action='store_true', default=False, help='The data to be used.')
parser.add_argument('-ld', '--load-data', action='store_true', default=False, help='Load preprocessed data.')
parser.add_argument('-kd', '--keep-duplicates', action='store_true', default=False, help='Do not remove duplicates in data.')
parser.add_argument('-g', '--gpu', action='store_true', default=False, help='Run on GPU(s).')
parser.add_argument('-lad', '--last-data', action='store_true', default=False, help='Use last working dataset.')


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
    results_masked = results.where(targets_mask, torch.tensor(-1, device=const.DEVICE))
    targets_masked = targets.where(targets_mask, torch.tensor(-1, device=const.DEVICE))
    return (results_masked == targets_masked).all(axis=1).sum().item()


@print_time()
def train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer, experiment, epoch):
    size = len(dataloader.dataset)
    current_batch_size = const.BATCH_SIZE
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    encoder.train()
    decoder.train()

    for batch, (inputs, targets, urls) in enumerate(dataloader):
        inputs, targets = inputs.to(const.DEVICE), targets.to(const.DEVICE)

        loss = 0
        output = []
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = inputs[0].size(0)
        target_length = targets[0].size(0)

        encoder_output, encoder_hidden = encoder(inputs)

        decoder_input = torch.tensor([[const.SOS_TOKEN] * current_batch_size], device=const.DEVICE)
        decoder_hidden = (encoder_hidden[0],
                          torch.zeros(const.BIDIRECTIONAL * const.ENCODER_LAYERS, current_batch_size, const.HIDDEN_SIZE,
                                      device=const.DEVICE))

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            output.append(topi.detach())

            current_target = targets[:, di].flatten()
            current_loss = loss_fn(decoder_output, current_target)

            if const.IGNORE_PADDING_IN_LOSS:
                loss_mask = current_target != const.PAD_TOKEN
                loss_masked = current_loss.where(loss_mask, torch.tensor(0.0, device=const.DEVICE))
                current_loss = loss_masked.sum() / loss_mask.sum() if loss_mask.sum() else 0
            loss += current_loss

        results = torch.cat(output).view(1, -1, current_batch_size).T
        accuracy = get_correct(results, targets) / current_batch_size

        loss /= target_length
        loss.backward()

        if const.GRADIENT_CLIPPING_ENABLED:
            nn.utils.clip_grad_norm_(encoder.parameters(),
                                     max_norm=const.GRADIENT_CLIPPING_MAX_NORM,
                                     norm_type=const.GRADIENT_CLIPPING_NORM_TYPE)
            nn.utils.clip_grad_norm_(decoder.parameters(),
                                     max_norm=const.GRADIENT_CLIPPING_MAX_NORM,
                                     norm_type=const.GRADIENT_CLIPPING_NORM_TYPE)

        experiment.log_train_metrics(loss.item(), get_grad_norm(encoder), get_grad_norm(encoder), input_length,
                                     accuracy,
                                     step=epoch * size / world_size / const.BATCH_SIZE + batch, epoch=epoch)
        encoder_optimizer.step()
        decoder_optimizer.step()


def test_loop(encoder, decoder, dataloader, loss_fn, experiment, epoch):
    test_loss, correct = 0, 0
    current_batch_size = const.BATCH_SIZE_TEST

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    size = len(dataloader.dataset) / world_size
    num_batches = int(size / current_batch_size)

    input_lang = dataloader.dataset.input_lang
    output_lang = dataloader.dataset.output_lang

    inputs = None

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for inputs, targets, urls in dataloader:
            inputs, targets = inputs.to(const.DEVICE), targets.to(const.DEVICE)

            target_length = targets[0].size(0)
            loss = 0
            output = []

            _, encoder_hidden = encoder(inputs)

            decoder_input = torch.tensor([[const.SOS_TOKEN] * current_batch_size], device=const.DEVICE)
            decoder_hidden = (encoder_hidden[0],
                              torch.zeros(const.BIDIRECTIONAL * const.ENCODER_LAYERS, current_batch_size,
                                          const.HIDDEN_SIZE,
                                          device=const.DEVICE))

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                output.append(topi.detach())

                current_target = targets[:, di].flatten()
                current_loss = loss_fn(decoder_output, current_target)

                if const.IGNORE_PADDING_IN_LOSS:
                    loss_mask = current_target != const.PAD_TOKEN
                    loss_masked = current_loss.where(loss_mask, torch.tensor(0.0, device=const.DEVICE))
                    current_loss = loss_masked.sum() / loss_mask.sum() if loss_mask.sum() else 0
                loss += current_loss

            test_loss += loss.item() / target_length
            # calc percentage of correctly generated sequences
            results = torch.cat(output).view(1, -1, current_batch_size).T
            correct += get_correct(results, targets)

    test_loss /= num_batches
    accuracy = correct / size

    experiment.log_test_metrics(input_lang, output_lang, inputs[:5], results[:5], test_loss, accuracy,
                                step=epoch, epoch=epoch)


def go_train(rank, world_size, experiment_name, port, train_data=None, test_data=None):
    if rank is not None:
        ddp.setup(rank, world_size, port)

    if not train_data:
        with open(const.DATA_WORKING_TRAIN_PATH, 'rb') as train_file:
            train_data = pickle.load(train_file)
    if not test_data:
        with open(const.DATA_WORKING_TEST_PATH, 'rb') as test_file:
            test_data = pickle.load(test_file)

    input_lang = train_data.input_lang
    output_lang = train_data.output_lang

    train_sampler = None
    test_sampler = None

    experiment = Experiment(experiment_name)
    experiment.log_initial_metrics(world_size, len(train_data), len(test_data), input_lang.n_words, output_lang.n_words)

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE, const.BATCH_SIZE, input_lang)
    decoder = model.DecoderRNN(const.HIDDEN_SIZE, output_lang.n_words, const.BATCH_SIZE, output_lang)

    if ddp.is_dist_avail_and_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=const.SHUFFLE_DATA)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=const.SHUFFLE_DATA)

        encoder = DistributedDataParallel(encoder.to(const.DEVICE), device_ids=[rank])
        decoder = DistributedDataParallel(decoder.to(const.DEVICE), device_ids=[rank])
    shuffle = const.SHUFFLE_DATA if (train_sampler is None) else None
    dataloader = loader.DataLoader(train_data, batch_size=const.BATCH_SIZE, shuffle=shuffle,
                                   collate_fn=pad_collate.PadCollate(), sampler=train_sampler, drop_last=True,
                                   num_workers=const.NUM_WORKERS_DATALOADER)
    test_dataloader = loader.DataLoader(test_data, batch_size=const.BATCH_SIZE, shuffle=shuffle,
                                        collate_fn=pad_collate.PadCollate(), sampler=test_sampler, drop_last=True,
                                        num_workers=const.NUM_WORKERS_DATALOADER)

    loss_fn = nn.NLLLoss(reduction='none') if const.IGNORE_PADDING_IN_LOSS else nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)

    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=const.LR_STEP_SIZE, gamma=const.LR_GAMMA)
    decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=const.LR_STEP_SIZE, gamma=const.LR_GAMMA)

    for epoch in range(const.EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        if train_sampler and test_sampler:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer, experiment, epoch)
        test_loop(encoder, decoder, test_dataloader, loss_fn, experiment, epoch)
        experiment.log_learning_rate(encoder_optimizer.param_groups[0]['lr'],
                                     decoder_optimizer.param_groups[0]['lr'], step=epoch, epoch=epoch)
        encoder_scheduler.step()
        decoder_scheduler.step()

    save(encoder, decoder)
    if ddp.is_dist_avail_and_initialized():
        ddp.cleanup()
    experiment.end()


def save(encoder, decoder):
    ddp.save_on_master(encoder.state_dict(), const.MODEL_ENCODER_PATH)
    ddp.save_on_master(decoder.state_dict(), const.MODEL_DECODER_PATH)

    print('saved models')


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
            go_train(None, 1, experiment_name, port, train_data, test_data)
        else:
            go_train(None, 1, experiment_name, port)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
