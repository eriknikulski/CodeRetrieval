import gzip
import json
import random

import const

BASE_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET_LENGTH = 100
SEQ_LENGTH = 5
N_TRAIN = 4000
N_TEST = 100000
N_VALID = 1000
TRAIN_PATH = const.PROJECT_PATH + 'data/synthetic/train/'
TEST_PATH = const.PROJECT_PATH + 'data/synthetic/test/'
VALID_PATH = const.PROJECT_PATH + 'data/synthetic/valid/'


def create_alphabet(base_alphabet, length):
    alphabet = ['']
    for i in range(length):
        alphabet.append(alphabet[int(i / len(base_alphabet))] + base_alphabet[i % len(base_alphabet)])
    return alphabet[1:]


def create_set(alphabet, length, n):
    samples = []
    while len(samples) < n:
        sample = random.sample(alphabet, length)
        if sample not in samples:
            samples.append(sample)
    return [{"docstring_tokens": sample, "code_tokens": random.sample(alphabet, length), "url": None} for sample in samples]


def create():
    alphabet = create_alphabet(BASE_ALPHABET, ALPHABET_LENGTH)
    samples = create_set(alphabet, SEQ_LENGTH, N_TRAIN + N_TEST + N_VALID)

    with gzip.open(TRAIN_PATH + 'synth_train.jsonl.gz', 'wb') as f:
        for item in samples[:N_TRAIN]:
            f.write((json.dumps(item) + '\n').encode('utf-8'))

    with gzip.open(TEST_PATH + 'synth_test.jsonl.gz', 'wb') as f:
        for item in samples[N_TRAIN:N_TRAIN + N_TEST]:
            f.write((json.dumps(item) + '\n').encode('utf-8'))

    with gzip.open(VALID_PATH + 'synth_valid.jsonl.gz', 'wb') as f:
        for item in samples[N_TEST:]:
            f.write((json.dumps(item) + '\n').encode('utf-8'))


if __name__ == "__main__":
    create()
