import re
import unicodedata

from bs4 import BeautifulSoup
import torch

import const


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {const.SOS_token: 'SOS', const.EOS_token: 'EOS'}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addSequence(self, seq):
        for word in seq:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indexesFromSequence(self, seq):
        return [self.word2index[word] for word in seq]

    def tensorFromSequence(self, seq):
        indexes = self.indexesFromSequence(seq)
        indexes.append(const.EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=const.DEVICE).view(-1, 1)

    def seqFromIndices(self, idcs):
        return [self.index2word[idx] for idx in idcs]

    def seqFromTensor(self, tensor):
        return self.seqFromIndices(el.item() for el in tensor)


def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeSeq(s):
    return [el.strip() for el in [normalizeString(el) for el in s] if el.strip()]


def normalizeDocstring(s):
    s = ' '.join(s).replace('< /', '</')
    return list(filter(None, BeautifulSoup(s).get_text().strip().split(' ')))


def normalizeString(s):
    s = unicode2Ascii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    return s
