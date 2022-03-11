import gzip
import json
import random

import const

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
LENGTH = 5
N_TRAIN = 4000
N_TEST = 100000
N_VALID = 1000
TRAIN_PATH = const.PROJECT_PATH + 'data/synthetic/train/'
TEST_PATH = const.PROJECT_PATH + 'data/synthetic/test/'
VALID_PATH = const.PROJECT_PATH + 'data/synthetic/valid/'


def create_seq(alphabet, length):
    doc = random.sample(alphabet, length)
    code = doc
    url = None
    return {"docstring_tokens": doc, "code_tokens": code, "url": url}


def create_set(alphabet, length, n):
    return [create_seq(alphabet, length) for _ in range(n)]


def create():
    with gzip.open(TRAIN_PATH + 'synth_train.jsonl.gz', 'wb') as f:
        for item in create_set(ALPHABET, LENGTH, N_TRAIN):
            f.write((json.dumps(item) + '\n').encode('utf-8'))

    with gzip.open(TEST_PATH + 'synth_test.jsonl.gz', 'wb') as f:
        for item in create_set(ALPHABET, LENGTH, N_TEST):
            f.write((json.dumps(item) + '\n').encode('utf-8'))

    with gzip.open(VALID_PATH + 'synth_valid.jsonl.gz', 'wb') as f:
        for item in create_set(ALPHABET, LENGTH, N_TEST):
            f.write((json.dumps(item) + '\n').encode('utf-8'))


if __name__ == "__main__":
    create()
