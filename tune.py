import os
import pickle

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
from torch import optim

import const
import loader
import model
import pad_collate
import train


def tune_train(config, checkpoint_dir=None):
    with open(const.DATA_WORKING_TRAIN_PATH, 'rb') as train_file:
        train_data = pickle.load(train_file)
    with open(const.DATA_WORKING_VALID_PATH, 'rb') as valid_file:
        valid_data = pickle.load(valid_file)

    input_lang = train_data.input_lang
    output_lang = train_data.output_lang

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE, const.BATCH_SIZE, input_lang)
    decoder = model.DecoderRNN(const.HIDDEN_SIZE, output_lang.n_words, const.BATCH_SIZE, output_lang)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            encoder = nn.DataParallel(encoder)
            decoder = nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)

    dataloader = loader.DataLoader(train_data, batch_size=int(config['batch_size']), shuffle=True,
                                   collate_fn=pad_collate.PadCollate(), drop_last=True,
                                   num_workers=const.NUM_WORKERS_DATALOADER)
    valid_dataloader = loader.DataLoader(valid_data, batch_size=int(config['batch_size']), shuffle=True,
                                         collate_fn=pad_collate.PadCollate(), drop_last=True,
                                         num_workers=const.NUM_WORKERS_DATALOADER)

    loss_fn = nn.NLLLoss(reduction='none') if const.IGNORE_PADDING_IN_LOSS else nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=config['lr'], momentum=const.MOMENTUM)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=config['lr'], momentum=const.MOMENTUM)

    if checkpoint_dir:
        encoder_state, decoder_state, encoder_optimizer_state, decoder_optimizer_state = \
            torch.load(os.path.join(checkpoint_dir, 'checkpoint'))
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(decoder_state)
        encoder_optimizer.load_state_dict(encoder_optimizer_state)
        decoder_optimizer.load_state_dict(decoder_optimizer_state)

    for epoch in range(10):
        train.train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer, None, epoch)
        valid_loss, valid_acc = train.valid_loop(encoder, decoder, valid_dataloader, loss_fn, None, epoch)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((encoder.state_dict(), decoder.state_dict(),
                        encoder_optimizer.state_dict(), decoder_optimizer.state_dict()), path)

        tune.report(loss=valid_loss, accuracy=valid_acc)
    print('Finished Training')


def run_test(best_trial, gpus_per_trial):
    with open(const.DATA_WORKING_TEST_PATH, 'rb') as valid_file:
        test_data = pickle.load(valid_file)

    input_lang = test_data.input_lang
    output_lang = test_data.output_lang

    best_encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE, const.BATCH_SIZE, input_lang)
    best_decoder = model.DecoderRNN(const.HIDDEN_SIZE, output_lang.n_words, const.BATCH_SIZE, output_lang)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if gpus_per_trial > 1:
            best_encoder = nn.DataParallel(best_encoder)
            best_decoder = nn.DataParallel(best_decoder)
    best_encoder.to(device)
    best_decoder.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    encoder_state, decoder_state, _, _ = torch.load(os.path.join(best_checkpoint_dir, 'checkpoint'))
    best_encoder.load_state_dict(encoder_state)
    best_decoder.load_state_dict(decoder_state)

    dataloader = loader.DataLoader(test_data, batch_size=int(best_trial.config['batch_size']), shuffle=True,
                                   collate_fn=pad_collate.PadCollate(), drop_last=True,
                                   num_workers=const.NUM_WORKERS_DATALOADER)

    loss_fn = nn.NLLLoss(reduction='none') if const.IGNORE_PADDING_IN_LOSS else nn.NLLLoss()

    test_loss, test_acc = train.test_loop(best_encoder, best_decoder, dataloader, loss_fn, experiment=None, epoch=0,
                                          device=device)
    print(f'Best trial test set accuracy: {test_acc}')
    return test_acc


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        'lr': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([64, 128, 256, 512])
    }
    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=['loss', 'accuracy', 'training_iteration'])
    result = tune.run(
        partial(tune_train),
        resources_per_trial={'cpu': 2, 'gpu': gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial('loss', 'min', 'last')
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')

    test_acc = run_test(best_trial, gpus_per_trial)
    print(f'Best trial test set accuracy: {test_acc}')


if __name__ == "__main__":
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
