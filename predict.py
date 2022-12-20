import csv
import pickle

import pandas as pd
import torch
from subword_nmt import subword_nmt

import const
import data
import model


def evaluate(encoder, decoder, sentence, max_length=const.MAX_LENGTH_CODE):
    with torch.no_grad():
        encoder_hidden = encoder.init_hidden()
        encoder_output, encoder_hidden = encoder(sentence, encoder_hidden)

        output = []

        decoder_input = torch.tensor([[const.SOS_TOKEN]], device=const.DEVICE)
        decoder_hidden = torch.cat(tuple(el for el in encoder_hidden[0]), dim=1).view(1, 1, -1)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            output.append(topi.item())
            decoder_input = topi.squeeze().detach()
            if topi.item() == 1:
                break
        return output


def preprocess(sentence, input_lang, bpe):
    sentence = data.normalize_docstring(sentence)
    sentence = bpe.segment_tokens(sentence)
    return input_lang.tensor_from_sequence(sentence)


def match(seq, df):
    res = df[df[['code_sequence']].applymap(lambda x: x == seq).values]
    if len(res):
        return res
    return pd.DataFrame({'url': None}, index=[0])


def run():
    # TODO: rework
    input_lang_file = open(const.DATA_INPUT_LANG_PATH, 'rb')
    input_lang = pickle.load(input_lang_file)
    output_lang_file = open(const.DATA_OUTPUT_LANG_PATH, 'rb')
    output_lang = pickle.load(output_lang_file)

    encoder = model.EncoderRNN(input_lang.n_words, const.HIDDEN_SIZE, 1, input_lang).to(const.DEVICE)
    encoder.load_state_dict(torch.load(const.ENCODER_PATH))
    encoder.eval()

    decoder = model.DecoderRNN(const.BIDIRECTIONAL * const.ENCODER_LAYERS * const.HIDDEN_SIZE,
                               output_lang.n_words, 1, output_lang).to(const.DEVICE)
    decoder.load_state_dict(torch.load(const.DECODER_PATH))
    decoder.eval()

    queries = pd.read_csv(const.QUERY_CSV_PATH)
    all_data = pd.read_pickle(const.DATA_ALL_DF_PATH)
    all_data[['code_sequence']] = all_data[['code_sequence']].applymap(lambda x: list(x.flatten()))

    with open(const.PREPROCESS_BPE_CODES_PATH, encoding='utf-8') as codes_file:
        bpe = subword_nmt.BPE(codes_file)

    predictions = []

    for i, row in queries.iterrows():
        sentence = preprocess(row['query'].split(), input_lang, bpe)
        output_seq = evaluate(encoder, decoder, sentence)
        output_sentence = ' '.join(decoder.lang.seq_from_indices(output_seq))
        output = match(output_seq, all_data)

        predictions.append(['Java', row['query'], output.at[0, 'url']])

    with open(const.MODEL_PREDICTIONS_CSV, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['language', 'query', 'url'])
        writer.writerows(predictions)


if __name__ == '__main__':
    run()
