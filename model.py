import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import const


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, lang):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lang = lang

        self.embedding = nn.Embedding(input_size, hidden_size).to(const.DEVICE)
        self.lstm = nn.LSTM(hidden_size, hidden_size, const.ENCODER_LAYERS,
                            bidirectional=True if const.BIDIRECTIONAL == 2 else False).to(const.DEVICE)

    def forward(self, input):
        rank = dist.get_rank() if dist.is_initialized() else None
        hidden = (torch.zeros(const.BIDIRECTIONAL * const.ENCODER_LAYERS, self.batch_size, self.hidden_size, device=const.DEVICE),
                  torch.zeros(const.BIDIRECTIONAL * const.ENCODER_LAYERS, self.batch_size, self.hidden_size, device=const.DEVICE))
        embedded = self.embedding(input).transpose(0, 1)
        output = embedded.view(-1, self.batch_size, self.hidden_size)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(const.BIDIRECTIONAL * const.ENCODER_LAYERS, self.batch_size, self.hidden_size, device=const.DEVICE), \
               torch.zeros(const.BIDIRECTIONAL * const.ENCODER_LAYERS, self.batch_size, self.hidden_size, device=const.DEVICE)

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, lang):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lang = lang

        self.embedding = nn.Embedding(output_size, hidden_size).to(const.DEVICE)
        self.lstm = nn.LSTM(hidden_size, hidden_size, const.DECODER_LAYERS,
                            bidirectional=True if const.BIDIRECTIONAL == 2 else False).to(const.DEVICE)
        self.out = nn.Linear(const.BIDIRECTIONAL * hidden_size, output_size).to(const.DEVICE)
        self.softmax = nn.LogSoftmax(dim=1).to(const.DEVICE)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, self.batch_size, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[-1]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(const.DECODER_LAYERS, self.batch_size, self.hidden_size, device=const.DEVICE),\
               torch.zeros(const.DECODER_LAYERS, self.batch_size, self.hidden_size, device=const.DEVICE)

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
