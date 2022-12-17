import re

import javalang
import unicodedata

from bs4 import BeautifulSoup
from dpu_utils.codeutils import split_identifier_into_parts
import torch

import const


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            const.SOS_TOKEN: 'SOS',
            const.EOS_TOKEN: 'EOS',
            const.PAD_TOKEN: 'PAD',
            const.OOV_TOKEN: 'OOV'}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_sequence(self, seq):
        for word in seq:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indexes_from_sequence(self, seq, oov_token=const.OOV_TOKEN):
        return [self.word2index[word] if word in self.word2index else oov_token for word in seq]

    def tensor_from_sequence(self, seq, oov_token=const.OOV_TOKEN):
        indexes = self.indexes_from_sequence(seq, oov_token)
        indexes.append(const.EOS_TOKEN)
        return torch.tensor(indexes, dtype=torch.long, device=const.DEVICE)

    def seq_from_indices(self, idcs):
        return [self.index2word[idx] for idx in idcs]

    def seq_from_tensor(self, tensor):
        return self.seq_from_indices(el.item() for el in tensor)

    def reduce_vocab(self, min_freq=None, max_tokens=None):
        assert min_freq or max_tokens
        if min_freq:
            for word in self.word2index.keys():
                if self.word2count[word] < min_freq:
                    del self.word2count[word]
        else:
            self.word2count = dict(sorted(self.word2count.items(), key=lambda item: item[1], reverse=True))
        self.rebuild_indices()

    def rebuild_indices(self):
        self.word2index = {}
        self.index2word = {
            const.SOS_TOKEN: 'SOS',
            const.EOS_TOKEN: 'EOS',
            const.PAD_TOKEN: 'PAD',
            const.OOV_TOKEN: 'OOV'}
        self.n_words = 4
        for word in self.word2count.keys():
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_seq(s):
    return [el.strip() for el in [normalize_string(el) for el in s] if el.strip()]


def normalize_docstring(s):
    if s[-1] == '.':
        s = s[:-1]

    s = ' '.join(s)
    s = split_identifier_into_parts(s)
    s = ' '.join(s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = s.replace('< /', '</')
    s = BeautifulSoup(s, features='html.parser').get_text().strip()
    # removes all doc annotations of the form { @someword }
    s = re.sub(r'\{\s?@\w+\s?}', r' ', s)
    # replace all doc annotations of the form { @someword _content } with _content
    s = re.sub(r'\{\s?@\w+\s([^}]*)\s?}', r'\1', s)
    # remove everything that has some form of 'non - javadoc' in it
    s = re.sub(r'.*\( non - javadoc \).*', r'', s)
    s = re.sub(r'@deprecated.*', r'', s)
    # remove everything that starts with 'to do'
    s = re.sub(r'^to\s?do.*', r'', s)
    # replace 'e . g .' with 'eg'
    s = re.sub(r'e \. g \.', r' eg ', s)
    s = re.sub(r'e \. g ', r' eg ', s)
    # replace 'i . e .' with 'ie'
    s = re.sub(r'i \. e \.', r' ie ', s)
    s = re.sub(r'i \. e ', r' ie ', s)
    # remove everything after first occurring dot including dot
    s = re.sub(r'\..*', r'', s)
    # remove all special chars but dot
    s = re.sub(r'[^\w\s.]', r' ', s)
    s = re.sub(r'\d+', f' {const.NUMBER_TOKEN} ', s)
    s = list(filter(None, s.split(' ')))
    return s


def normalize_code(s):
    RE_WORDS = re.compile(r'''
            # Find words in a string. Order matters!
            [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
            [A-Z]?[a-z]+ |  # Capitalized words / all lower case
            [A-Z]+ |  # All upper case
            \d+ | # Numbers
            .+
        ''', re.VERBOSE)

    def split_subtokens(s):
        return [sub_token for sub_token in RE_WORDS.findall(s) if not sub_token == '_']

    modifiers = ['public', 'private', 'protected', 'static']
    s = s.encode('ascii', 'ignore').decode('ascii')
    # remove annotations
    s = re.sub(r'^\s*@.*', r'', s)

    tokens = list(javalang.tokenizer.tokenize(s))
    s = ' '.join([' '.join(split_subtokens(i.value)) for i in tokens if i.value not in modifiers])

    # replace text
    s = re.sub(r'""".*"""', const.TEXT_TOKEN, s)
    s = re.sub(r'".*"', const.TEXT_TOKEN, s)
    s = re.sub(r'\'.*\'', const.CHAR_TOKEN, s)
    s = list(filter(None, s.split(' ')))
    return s


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'[^a-zA-Z]+', r' ', s)
    return s
