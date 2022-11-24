import re
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

    def indexesFromSequence(self, seq, oov_token=const.OOV_TOKEN):
        return [self.word2index[word] if word in self.word2index else oov_token for word in seq]

    def tensorFromSequence(self, seq, oov_token=const.OOV_TOKEN):
        indexes = self.indexesFromSequence(seq, oov_token)
        indexes.append(const.EOS_TOKEN)
        return torch.tensor(indexes, dtype=torch.long, device=const.DEVICE).view(-1, 1)

    def seqFromIndices(self, idcs):
        return [self.index2word[idx] for idx in idcs]

    def seqFromTensor(self, tensor):
        return self.seqFromIndices(el.item() for el in tensor)

    def reduceVocab(self, min_freq):
        for word, index in self.word2index.items():
            if self.word2count[word] < min_freq:
                del self.word2count[word]
                del self.word2index[word]
                del self.index2word[index]
                self.n_words -= 1


def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeSeq(s):
    return [el.strip() for el in [normalizeString(el) for el in s] if el.strip()]


def normalizeDocstring(s):
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


def normalizeString(s):
    s = unicode2Ascii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    return s
