from __future__ import unicode_literals, print_function, division
import argparse
import hashlib
import os
import pickle
import random
import string
import time
import math

import comet_ml
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel

import const
import data
import ddp
import keys
import loader
import model
import pad_collate


parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-d', '--data', choices=['java', 'synth'], help='The data to be used.')
parser.add_argument('-lo', '--labels-only', action='store_true', default=False, help='The data to be used.')
parser.add_argument('-ld', '--load-data', action='store_true', default=False, help='Load preprocessed data.')
parser.add_argument('-kd', '--keep-duplicates', action='store_true', default=False, help='Do not remove duplicates in data.')
parser.add_argument('-g', '--gpu', action='store_true', default=False, help='Run on GPU(s).')


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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def getGradientNorm(model):
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) > 0:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item()
    return total_norm


@print_time()
def train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer, rank, experiment, epoch_num):
    size = len(dataloader.dataset)
    current_batch_size = const.BATCH_SIZE
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    experiment.log_metric(f'training_set_size', size)

    encoder.train()
    decoder.train()

    for batch, (inputs, targets, urls) in enumerate(dataloader):
        inputs, targets = inputs.to(const.DEVICE), targets.to(const.DEVICE)

        loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = inputs[0].size(0)
        target_length = targets[0].size(0)

        experiment.log_metric(f'seq_length', input_length,
                              step=epoch_num * size / world_size / const.BATCH_SIZE + batch)

        encoder_output, encoder_hidden = encoder(inputs)

        decoder_input = torch.tensor([[const.SOS_TOKEN] * current_batch_size], device=const.DEVICE)
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            current_target = targets[:, di].flatten()
            current_loss = loss_fn(decoder_output, current_target)

            if const.IGNORE_PADDING_IN_LOSS:
                loss_mask = current_target != const.PAD_TOKEN
                loss_masked = current_loss.where(loss_mask, torch.tensor(0.0, device=const.DEVICE))
                current_loss = loss_masked.sum() / loss_mask.sum() if loss_mask.sum() else 0
            loss += current_loss

        loss = loss / target_length
        loss.backward()

        if const.GRADIENT_CLIPPING_ENABLED:
            nn.utils.clip_grad_norm_(encoder.parameters(),
                                     max_norm=const.GRADIENT_CLIPPING_MAX_NORM, norm_type=const.GRADIENT_CLIPPING_NORM_TYPE)
            nn.utils.clip_grad_norm_(decoder.parameters(),
                                     max_norm=const.GRADIENT_CLIPPING_MAX_NORM, norm_type=const.GRADIENT_CLIPPING_NORM_TYPE)

        if rank:
            experiment.log_metric(f'{rank}_batch_loss', loss.item(),
                                  step=epoch_num * size / world_size / const.BATCH_SIZE + batch)
            experiment.log_metric(f'{rank}_encoder_grad_norm', getGradientNorm(encoder),
                                  step=epoch_num * size / world_size / const.BATCH_SIZE + batch)
            experiment.log_metric(f'{rank}_decoder_grad_norm', getGradientNorm(decoder),
                                  step=epoch_num * size / world_size / const.BATCH_SIZE + batch)
        else:
            experiment.log_metric(f'batch_loss', loss.item(), step=epoch_num * size / const.BATCH_SIZE + batch)
            experiment.log_metric(f'encoder_grad_norm', getGradientNorm(encoder),
                                  step=epoch_num * size / world_size / const.BATCH_SIZE + batch)
            experiment.log_metric(f'decoder_grad_norm', getGradientNorm(decoder),
                                  step=epoch_num * size / world_size / const.BATCH_SIZE + batch)
        encoder_optimizer.step()
        decoder_optimizer.step()


def test_loop(encoder, decoder, dataloader, loss_fn, rank, experiment, epoch_num):
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()
        encoder = encoder.module
        decoder = decoder.module
    size = len(dataloader.dataset) / world_size
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    current_batch_size = const.BATCH_SIZE_TEST
    input_lang = dataloader.dataset.input_lang
    output_lang = dataloader.dataset.output_lang

    inputs = None

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for inputs, targets, urls in dataloader:
            inputs, targets = inputs.to(const.DEVICE), targets.to(const.DEVICE)

            input_length = inputs[0].size(0)
            target_length = targets[0].size(0)

            loss = 0
            output = []

            _, encoder_hidden = encoder(inputs)

            decoder_input = torch.tensor([[const.SOS_TOKEN] * current_batch_size], device=const.DEVICE)
            decoder_hidden = encoder_hidden

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
            results = torch.cat(output).view(1, -1, current_batch_size).T
            targets_mask = targets != const.PAD_TOKEN
            results_masked = results.where(targets_mask, torch.tensor(-1, device=const.DEVICE))
            targets_masked = targets.where(targets_mask, torch.tensor(-1, device=const.DEVICE))
            correct += (results_masked.to(rank) == targets_masked.to(rank)).all(axis=1).sum().item()

    inputs = [' '.join(input_lang.seqFromTensor(el.flatten())) for el in inputs[:5]]
    results = [' '.join(output_lang.seqFromTensor(el.flatten())) for el in results[:5]]
    experiment.log_text(str(epoch_num) + '\n' +
                        '\n\n'.join(str(input) + '\n  ====>  \n' + str(result) for input, result in zip(inputs, results)))

    test_loss /= num_batches
    correct /= size
    if rank:
        experiment.log_metric(f'{rank}_test_batch_loss', test_loss, step=epoch_num)
        experiment.log_metric(f'{rank}_accuracy', 100 * correct, step=epoch_num)
    else:
        experiment.log_metric(f'test_batch_loss', test_loss, step=epoch_num)
        experiment.log_metric(f'accuracy', 100 * correct, step=epoch_num)


def get_experiment(run_id):
    experiment_key = hashlib.sha1(run_id.encode('utf-8')).hexdigest()
    os.environ['COMET_EXPERIMENT_KEY'] = experiment_key

    api = comet_ml.API(api_key=keys.COMET_API_KEY)
    api_experiment = api.get_experiment_by_key(experiment_key)

    if not api_experiment:
        return comet_ml.Experiment(
            api_key=keys.COMET_API_KEY,
            project_name=const.COMET_PROJECT_NAME,
            workspace=const.COMET_WORKSPACE,)
    else:
        return comet_ml.ExistingExperiment(
            api_key=keys.COMET_API_KEY,
            project_name=const.COMET_PROJECT_NAME,
            workspace=const.COMET_WORKSPACE,)


def go_train(rank, world_size, train_data, test_data, experiment_name, port):
    input_lang = train_data.input_lang
    output_lang = train_data.output_lang

    train_sampler = None
    test_sampler = None

    experiment = get_experiment(experiment_name)
    experiment.log_parameters(const.HYPER_PARAMS)
    experiment.log_parameter('world_size', world_size)
    experiment.log_parameter('train_data_size', len(train_data))
    experiment.log_parameter('test_data_size', len(test_data))
    experiment.log_parameter('input_lang_n_words', input_lang.n_words)
    experiment.log_parameter('output_lang_n_words', output_lang.n_words)

    if rank is not None:
        ddp.setup(rank, world_size, port)

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE, const.BATCH_SIZE, input_lang)
    decoder = model.DecoderRNN(const.HIDDEN_SIZE, output_lang.n_words, const.BATCH_SIZE, output_lang)

    if rank is not None:
        experiment.log_parameter('port', os.environ['MASTER_PORT'])

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=const.SHUFFLE_DATA)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=const.SHUFFLE_DATA)

        encoder = DistributedDataParallel(encoder.to(const.DEVICE), device_ids=[rank])
        decoder = DistributedDataParallel(decoder.to(const.DEVICE), device_ids=[rank])

    dataloader = loader.DataLoader(train_data, batch_size=const.BATCH_SIZE, shuffle=const.SHUFFLE_DATA,
                                   collate_fn=pad_collate.PadCollate(), sampler=train_sampler, drop_last=True,
                                   num_workers=const.NUM_WORKERS_DATALOADER)
    test_dataloader = loader.DataLoader(test_data, batch_size=const.BATCH_SIZE, shuffle=const.SHUFFLE_DATA,
                                        collate_fn=pad_collate.PadCollate(), sampler=test_sampler, drop_last=True,
                                        num_workers=const.NUM_WORKERS_DATALOADER)

    loss_fn = nn.NLLLoss(reduction='none') if const.IGNORE_PADDING_IN_LOSS else nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)

    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=const.LR_STEP_SIZE, gamma=const.LR_GAMMA)
    decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=const.LR_STEP_SIZE, gamma=const.LR_GAMMA)

    with experiment.train():
        for epoch in range(const.EPOCHS):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            if train_sampler and test_sampler:
                train_sampler.set_epoch(epoch)
                test_sampler.set_epoch(epoch)
            train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer,
                       rank, experiment, epoch)
            with experiment.test():
                test_loop(encoder, decoder, test_dataloader, loss_fn, rank, experiment, epoch)
            encoder_scheduler.step()
            decoder_scheduler.step()

            if rank:
                experiment.log_metric(f'{rank}_learning_rate_encoder', encoder_optimizer.param_groups[0]['lr'],
                                      step=epoch)
                experiment.log_metric(f'{rank}_learning_rate_decoder', decoder_optimizer.param_groups[0]['lr'],
                                      step=epoch)
            else:
                experiment.log_metric(f'learning_rate_encoder', encoder_optimizer.param_groups[0]['lr'], step=epoch)
                experiment.log_metric(f'learning_rate_decoder', decoder_optimizer.param_groups[0]['lr'], step=epoch)

    if not rank:
        save(encoder, decoder)
    if rank is not None:
        ddp.cleanup()
    experiment.end()


def save(encoder, decoder):
    torch.save(encoder.state_dict(), const.ENCODER_PATH)
    torch.save(decoder.state_dict(), const.DECODER_PATH)

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

    if args.load_data:
        train_file = open(const.TRAIN_DATA_SAVE_PATH, 'rb')
        train_data = pickle.load(train_file)
        train_data.enforce_length_constraints()
        test_file = open(const.TEST_DATA_SAVE_PATH, 'rb')
        test_data = pickle.load(test_file)
        test_data.enforce_length_constraints()
        valid_file = open(const.VALID_DATA_SAVE_PATH, 'rb')
        valid_data = pickle.load(valid_file)
        valid_data.enforce_length_constraints()

        input_lang_file = open(const.INPUT_LANG_SAVE_PATH, 'rb')
        input_lang = pickle.load(input_lang_file)
        output_lang_file = open(const.OUTPUT_LANG_SAVE_PATH, 'rb')
        output_lang = pickle.load(output_lang_file)

        if const.LABELS_ONLY:
            train_data.df['code_tokens'] = train_data.df['docstring_tokens']
            test_data.df['code_tokens'] = test_data.df['docstring_tokens']
            valid_data.df['code_tokens'] = valid_data.df['docstring_tokens']

            output_lang = input_lang

            train_data.output_lang = train_data.input_lang
            test_data.output_lang = test_data.input_lang
            valid_data.output_lang = valid_data.input_lang

            train_data.sort()
            test_data.sort()
            valid_data.sort()

            train_data.to_numpy()
            test_data.to_numpy()
            valid_data.to_numpy()
    else:
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

    const.HYPER_PARAMS['input_lang.n_words'] = input_lang.n_words
    const.HYPER_PARAMS['output_lang.n_words'] = output_lang.n_words

    experiment_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(const.COMET_EXP_NAME_LENGTH))
    port = ddp.find_free_port(const.MASTER_ADDR)
    print(f'CUDA_DEVICE_COUNT: {const.CUDA_DEVICE_COUNT}')
    if args.gpu:
        ddp.run(go_train, const.CUDA_DEVICE_COUNT, train_data, test_data, experiment_name, port)
    else:
        go_train(None, 1, train_data, test_data, experiment_name, port)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
