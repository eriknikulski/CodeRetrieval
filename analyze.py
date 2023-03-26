import argparse
import collections
import pickle

from subword_nmt import learn_bpe
from matplotlib import pyplot as plt
from nltk.util import ngrams

import const
from loader import CodeDataset


def analyze_vocab(vocab):
    print('Creating train data vocab histogram...')
    num_bins = const.MAX_LENGTH_DOCSTRING
    range_h = [const.MIN_LENGTH_DOCSTRING, const.MAX_LENGTH_DOCSTRING]
    x = vocab.values()
    plt.figure()
    n, bins, patches = plt.hist(x, num_bins, range=range_h, facecolor='blue', alpha=0.5)
    plt.title('Vocabulary term occurrence')
    plt.xlabel('term length')
    plt.ylabel('frequency')
    plt.savefig(const.ANALYZE_VOCAB_HISTOGRAM)

    print(f'Train data vocab is of size: {len(vocab)}')


def analyze_entries(entries, title=None, save_path=const.ANALYZE_DATA_HISTOGRAM):
    print('Creating train data histogram...')
    num_bins = const.MAX_LENGTH_DOCSTRING
    range_h = [const.MIN_LENGTH_DOCSTRING, const.MAX_LENGTH_DOCSTRING]
    plt.figure()
    n, bins, patches = plt.hist(entries, num_bins, range=range_h, facecolor='blue', alpha=0.5)
    plt.title(title)
    plt.xlabel('sequence length')
    plt.ylabel('frequency')
    plt.savefig(save_path)

    print(f'Train data is of size: {len(entries)}')


def analyze_vocab_dataset(file_path=None):
    if file_path:
        print('Reading train file...')
        with open(file_path, 'rb') as train_file:
            train_data = pickle.load(train_file)
            train_data.enforce_length_constraints()
    else:
        train_data = CodeDataset(const.PROJECT_PATH + const.JAVA_PATH + 'train/',
                                 to_tensors=False)

    analyze_vocab(train_data.lang.word2count)
    analyze_entries(train_data.df['docstring_tokens'].map(len).to_list(),
                    title='Docstring sequence length', save_path=const.ANALYZE_PATH + 'doc_hist.png')
    analyze_entries(train_data.df['code_sequence'].map(len).to_list(),
                    title='Code sequence length', save_path=const.ANALYZE_PATH + 'code_hist.png')


def analyze_vocab_train_file(train_file_path=const.PREPROCESS_BPE_TRAIN_PATH):
    print('Reading train file...')
    with open(train_file_path, encoding='utf-8') as train_file:
        vocab = learn_bpe.get_vocabulary(train_file)
        analyze_vocab(vocab)

    lines = []
    with open(train_file_path, encoding='utf-8') as train_file:
        for i, line in enumerate(train_file):
            lines.append(len(line.strip('\r\n ').split(' ')))
        analyze_entries(lines)

def get_ngram_occurrence(file, ngram_size=2, lower_freq_limit=10):
    with open(file, 'r') as f:
        grams = ngrams(f.read().split(), ngram_size)

    return [(elem, count) for elem, count in collections.Counter(grams).most_common() if count >= lower_freq_limit]


def code_seq_occurrence(lower, upper):
    code_seq_file = const.DATA_PATH + 'code_sequences.csv'
    code_seq_occurrence_file = const.ANALYZE_OCCURRENCE + 'code_sequences'

    for i in range(lower, upper):
        res = get_ngram_occurrence(code_seq_file, ngram_size=i)

        with open(f'{code_seq_occurrence_file}_{i}.csv', 'w') as f:
            f.writelines([f'{occ} : {" ".join(item)}\n' for item, occ in res])


def code_tokens_occurrence(lower, upper):
    code_tokens_file = const.DATA_PATH + 'code_tokens.csv'
    code_tokens_occurrence_file = const.ANALYZE_OCCURRENCE + 'code_tokens'

    for i in range(lower, upper):
        res = get_ngram_occurrence(code_tokens_file, ngram_size=i)

        with open(f'{code_tokens_occurrence_file}_{i}.csv', 'w') as f:
            f.writelines([f'{occ} : {" ".join(item)}\n' for item, occ in res])


def methode_name_occurrence(lower, upper):
    methode_name_file = const.DATA_PATH + 'methode_name.csv'
    methode_name_occurrence_file = const.ANALYZE_OCCURRENCE + 'methode_name'

    for i in range(lower, upper):
        res = get_ngram_occurrence(methode_name_file, ngram_size=i)

        with open(f'{methode_name_occurrence_file}_{i}.csv', 'w') as f:
            f.writelines([f'{occ} : {" ".join(item)}\n' for item, occ in res])


parser = argparse.ArgumentParser(description='ML model for sequence to sequence translation')
parser.add_argument('-task', '--task', choices=['vocab', 'ngram'], help='What to analyze')
parser.add_argument('-t', '--type', choices=['dataset', 'file'], help='Chose either dataset or file. When task is vocab.')
parser.add_argument('-p', '--file-path', help='The file path to be used if type is file')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.task == 'ngram':
        code_seq_occurrence(lower=2, upper=20)
        code_tokens_occurrence(lower=2, upper=20)
        methode_name_occurrence(lower=2, upper=20)
    else:
        if args.type == 'dataset':
            analyze_vocab_dataset()
        else:
            analyze_vocab_train_file(args.file_path)
