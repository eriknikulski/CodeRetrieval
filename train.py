from __future__ import unicode_literals, print_function, division
import time
import math

import torch
import torch.nn as nn
from torch import optim

import const
import loader
import model

# plt.switch_backend('agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer, max_length=const.MAX_LENGTH):
    size = len(dataloader.dataset)
    for batch, ([input], [target], [url]) in enumerate(dataloader):
        # Compute prediction and loss
        encoder_hidden = encoder.initHidden()

        input_length = input.size(0)
        target_length = target.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[const.SOS_token]], device=device)
        decoder_hidden = encoder_hidden[0]

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += loss_fn(decoder_output, target[di])
            if decoder_input.item() == const.EOS_token:
                break

        # Backpropagation
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item() / target_length, batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(encoder, decoder, dataloader, loss_fn, max_length=const.MAX_LENGTH):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for [input], [target], [url] in dataloader:
            encoder_hidden = encoder.initHidden()

            input_length = input.size(0)
            target_length = target.size(0)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            loss = 0
            output = []

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[const.SOS_token]], device=device)
            decoder_hidden = encoder_hidden[0]

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                output.append(topi)
                loss += loss_fn(decoder_output, target[di])
                if decoder_input.item() == const.EOS_token:
                    break

            test_loss += loss.item() / target_length
            correct += int(output == input)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def go_train(encoder, decoder, dataloader, test_dataloader, epochs=const.EPOCHS):
    loss_fn = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=const.LEARNING_RATE, momentum=const.MOMENTUM)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(encoder, decoder, dataloader, loss_fn, encoder_optimizer, decoder_optimizer)
        test_loop(encoder, decoder, test_dataloader, loss_fn)
    print("Done!")
    print(f'LR: {const.LEARNING_RATE}')


def run():
    # train_data = loader.CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'train/', only_labels=True)
    # train_dataloader = loader.DataLoader(train_data, batch_size=1, shuffle=True)
    train_dataloader = None

    test_data = loader.CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'test/', only_labels=True)
    test_dataloader = loader.DataLoader(test_data, batch_size=1, shuffle=True)

    # valid_data = loader.CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'valid/', only_labels=True)
    # valid_dataloader = loader.DataLoader(valid_data, batch_size=1, shuffle=True)

    # input_lang, output_lang = train_data.get_langs()
    input_lang, output_lang = test_data.get_langs()

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE).to(device)
    decoder = model.DecoderRNN(const.HIDDEN_SIZE, output_lang.n_words).to(device)

    go_train(encoder, decoder, test_dataloader, test_dataloader)

    # torch.save(encoder.state_dict(), const.ENCODER_PATH)
    # torch.save(decoder.state_dict(), const.DECODER_PATH)
    #
    # print('saved models')


if __name__ == '__main__':
    run()
