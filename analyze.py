import argparse
import pickle

from subword_nmt import learn_bpe
from matplotlib import pyplot as plt

import const


def analyze_vocab(vocab):
    print('Creating train data vocab histogram...')
    num_bins = const.MAX_LENGTH_DOCSTRING
    range_h = [const.MIN_LENGTH_DOCSTRING, const.MAX_LENGTH_DOCSTRING]
    x = vocab.values()
    plt.figure()
    n, bins, patches = plt.hist(x, num_bins, range=range_h, facecolor='blue', alpha=0.5)
    plt.savefig(const.ANALYZE_VOCAB_HISTOGRAM)

    print(f'Train data vocab is of size: {len(vocab)}')


def analyze_entries(entries):
    print('Creating train data histogram...')
    num_bins = const.MAX_LENGTH_DOCSTRING
    range_h = [const.MIN_LENGTH_DOCSTRING, const.MAX_LENGTH_DOCSTRING]
    plt.figure()
    n, bins, patches = plt.hist(entries, num_bins, range=range_h, facecolor='blue', alpha=0.5)
    plt.savefig(const.ANALYZE_DATA_HISTOGRAM)

    print(f'Train data is of size: {len(entries)}')


def analyze_vocab_dataset():
    print('Reading train file...')
    with open(const.DATA_TRAIN_PATH, 'rb') as train_file:
        train_data = pickle.load(train_file)
        train_data.enforce_length_constraints()

    analyze_vocab(train_data.input_lang.word2count)
    analyze_entries(train_data.df['docstring_tokens'].map(len).to_numpy())


def analyze_vocab_train_file(train_file_path=const.PREPROCESS_BPE_TRAIN_PATH_DOC):
    print('Reading train file...')
    with open(train_file_path, encoding='utf-8') as train_file:
        vocab = learn_bpe.get_vocabulary(train_file)
        analyze_vocab(vocab)

    lines = []
    with open(train_file_path, encoding='utf-8') as train_file:
        for i, line in enumerate(train_file):
            lines.append(len(line.strip('\r\n ').split(' ')))
        analyze_entries(lines)


parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-t', '--type', choices=['dataset', 'file'], help='Chose either dataset or file')
parser.add_argument('-p', '--file-path', help='The file path to be used if type is file')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.type == 'dataset':
        analyze_vocab_dataset()
    else:
        analyze_vocab_train_file(args.file_path)
