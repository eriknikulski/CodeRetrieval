from __future__ import unicode_literals, print_function, division

import random

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import const
import data
import model
import loader

# plt.switch_backend('agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=const.MAX_LENGTH):
    with torch.no_grad():
        input_tensor = data.tensorFromSequence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[const.SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == const.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence, pairs, encoder, decoder, input_lang, output_lang):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
    sim = findMostSim(output_lang, output_words, pairs)
    print(sim)


def cosineSim(v1, v2):
    return np.dot(v1, v2) / (np.norm(v1) * np.norm(v2))


def bagSim(v1, v2):
    if len(v2) > len(v1):
        tmp = v2
        v2 = v1
        v1 = tmp
    return sum(1 for word in v1 if word in v2)


def findMostSim(output_lang, words, pairs):
    words = words[:-1]
    vec = data.tensorFromSequence(output_lang, words)
    result = None
    sMax = 0
    for pair in pairs:
        sim = bagSim(vec, data.tensorFromSequence(output_lang, pair[1]))
        if sim > sMax:
            sMax = sim
            result = pair[1]

    return result


def run():
    input_lang, output_lang, pairs = loader.get()

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE).to(device)
    encoder.load_state_dict(torch.load(const.SAVE_PATH + 'encoder.pt'))
    encoder.eval()

    decoder = model.AttnDecoderRNN(const.HIDDEN_SIZE, output_lang.n_words, dropout_p=0.1).to(device)
    decoder.load_state_dict(torch.load(const.SAVE_PATH + 'decoder.pt'))
    decoder.eval()

    for i in range(4):
        pair = random.choice(pairs)
        print('\nOriginal:')
        print('docstring: ' + ' '.join(pair[0]))
        print(pair[1])
        print('Infer')
        evaluateAndShowAttention(pair[0], pairs, encoder, decoder, input_lang, output_lang)


if __name__ == '__main__':
    run()
