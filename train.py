from __future__ import unicode_literals, print_function, division
from datetime import datetime
import time
import math

import torch
import torch.nn as nn
from matplotlib import pyplot as plt, ticker
from torch import optim

import const
import loader
import model
import pad_collate

# plt.switch_backend('agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@print_time()
def train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer, max_length=const.MAX_LENGTH):
    losses = []
    size = len(dataloader.dataset)
    current_batch_size = const.BATCH_SIZE
    encoder.setBatchSize(current_batch_size)
    decoder.setBatchSize(current_batch_size)

    for batch, (inputs, targets, urls) in enumerate(dataloader):
        if len(inputs) < const.BATCH_SIZE:
            current_batch_size = len(inputs)
            encoder.setBatchSize(current_batch_size)
            decoder.setBatchSize(current_batch_size)

        loss = 0

        input_length = inputs[0].size(0)
        target_length = targets[0].size(0)

        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_output, encoder_hidden = encoder(inputs, encoder_hidden)
        encoder_outputs = encoder_output[0, 0]

        decoder_input = torch.tensor([[const.SOS_token] * current_batch_size], device=device)
        decoder_hidden = encoder_hidden[0]

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += loss_fn(decoder_output, targets[:,di].flatten())
        loss = loss / target_length

        # Backpropagation
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        losses.append(loss.item())

        if batch % const.TRAINING_PER_BATCH_PRINT == 0:
            loss, current = loss.item(), batch * const.BATCH_SIZE
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses


def test_loop(encoder, decoder, dataloader, loss_fn, max_length=const.MAX_LENGTH):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    current_batch_size = const.BATCH_SIZE_TEST
    encoder.setBatchSize(current_batch_size)
    decoder.setBatchSize(current_batch_size)

    with torch.no_grad():
        for inputs, targets, urls in dataloader:
            if len(inputs) < const.BATCH_SIZE_TEST:
                current_batch_size = len(inputs)
                encoder.setBatchSize(current_batch_size)
                decoder.setBatchSize(current_batch_size)
            encoder_hidden = encoder.initHidden()

            input_length = inputs[0].size(0)
            target_length = targets[0].size(0)

            loss = 0
            output = []

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
            encoder_output, encoder_hidden = encoder(inputs, encoder_hidden)
            encoder_outputs = encoder_output[0, 0]

            decoder_input = torch.tensor([[const.SOS_token] * current_batch_size], device=device)
            decoder_hidden = encoder_hidden[0]

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                output.append(topi)
                loss += loss_fn(decoder_output, targets[:, di].flatten())

            test_loss += loss.item() / target_length
            correct += (torch.cat(output).view(1, -1, current_batch_size).T == targets).all(axis=1).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


@print_time('\nTotal ')
def go_train(encoder, decoder, dataloader, test_dataloader, epochs=const.EPOCHS):
    losses_train = []
    losses_test = []
    loss_fn = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        losses_train.extend(train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer))
        losses_test.append(test_loop(encoder, decoder, test_dataloader, loss_fn))
    showPlot(losses_train, 'train')
    showPlot(losses_test, 'test')
    print("Done!")
    print(f'LR: {const.LEARNING_RATE}')


def showPlot(points, descriptor=''):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.05)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    fig.show()
    plt.savefig(const.LOSS_PLOT_PATH + descriptor + '_loss_plot_lr_' + str(const.LEARNING_RATE).replace('.', '_') + '_'
                + str(const.EPOCHS) + 'epochs_' + str(datetime.now()) + '.png')


def run():
    # train_data = loader.CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'train/', only_labels=True)
    # train_dataloader = loader.DataLoader(train_data, batch_size=1, shuffle=True)
    train_dataloader = None

    test_data = loader.CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'test/', only_labels=True)
    test_dataloader = loader.DataLoader(test_data, batch_size=const.BATCH_SIZE, shuffle=True,
                                        collate_fn=pad_collate.PadCollate())

    valid_data = loader.CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'valid/', only_labels=True)
    valid_dataloader = loader.DataLoader(valid_data, batch_size=const.BATCH_SIZE_TEST, shuffle=True,
                                         collate_fn=pad_collate.PadCollate())

    # input_lang, output_lang = train_data.get_langs()
    input_lang, output_lang = test_data.get_langs()

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE, const.BATCH_SIZE).to(device)
    decoder = model.DecoderRNN(const.HIDDEN_SIZE, output_lang.n_words, const.BATCH_SIZE).to(device)

    go_train(encoder, decoder, test_dataloader, valid_dataloader)

    # torch.save(encoder.state_dict(), const.ENCODER_PATH)
    # torch.save(decoder.state_dict(), const.DECODER_PATH)
    #
    # print('saved models')


if __name__ == '__main__':
    run()
