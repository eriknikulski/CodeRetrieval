from __future__ import unicode_literals, print_function, division

import numpy as np
import pandas as pd
import torch

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

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == const.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def cosineSim(v1, v2):
    return np.dot(v1, v2) / (np.norm(v1) * np.norm(v2))


def bagSim(v1, v2):
    if len(v2) > len(v1):
        tmp = v2
        v2 = v1
        v1 = tmp
    return sum(1 for word in v1 if word in v2) / len(v1)


def findMostSim(output_lang, words, pairs, func=bagSim):
    words = words[:-1]
    vec = data.tensorFromSequence(output_lang, words)
    result = None
    sMax = 0
    for pair in pairs:
        sim = func(vec, data.tensorFromSequence(output_lang, pair[1]))
        if sim > sMax:
            sMax = sim
            result = pair

    return result


def run():
    input_lang, output_lang, pairs = loader.get()

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE).to(device)
    encoder.load_state_dict(torch.load(const.SAVE_PATH + 'encoder.pt'))
    encoder.eval()

    decoder = model.DecoderRNN(const.HIDDEN_SIZE, output_lang.n_words).to(device)
    decoder.load_state_dict(torch.load(const.SAVE_PATH + 'decoder.pt'))
    decoder.eval()

    queries = pd.read_csv(const.QUERY_CSV_PATH)

    for i, row in queries.iterrows():
        print()
        print()
        print(row['query'])
        output_words = evaluate(encoder, decoder, row['query'].split(), input_lang, output_lang)
        found = [pair for pair in pairs if output_words == pairs[1]]
        sim = findMostSim(output_lang, output_words, pairs)
        print(output_words)
        print(found)
        print(sim)
        if i > 5:
            break


if __name__ == '__main__':
    run()
