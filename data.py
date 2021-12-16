import re
import unicodedata
import torch

import const

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeSeq(s):
    val = [el.strip().lower() for el in [normalizeString(el) for el in s] if el.strip()]
    return val


def normalizeString(s):
    s = unicode2Ascii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"([.!?])", r"", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r"", s)
    return s


def indexesFromSequence(lang, seq):
    return [lang.word2index[word] for word in seq]


def tensorFromSequence(lang, seq):
    indexes = indexesFromSequence(lang, seq)
    indexes.append(const.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromSeqPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSequence(input_lang, pair[0])
    target_tensor = tensorFromSequence(output_lang, pair[1])
    return input_tensor, target_tensor
