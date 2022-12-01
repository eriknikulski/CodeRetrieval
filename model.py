import torch
import torch.nn as nn
import torch.nn.functional as F

import const


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, lang, bidirectional=const.BIDIRECTIONAL, 
                 layers=const.ENCODER_LAYERS, dropout=const.LSTM_ENCODER_DROPOUT, device=const.DEVICE):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lang = lang
        self.bidirectional = bidirectional
        self.layers = layers
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size).to(self.device)
        self.lstm = nn.LSTM(hidden_size, hidden_size, self.layers,
                            bidirectional=True if self.bidirectional == 2 else False,
                            dropout=self.dropout).to(self.device)

    def forward(self, input):
        hidden = (torch.zeros(self.bidirectional * self.layers, self.batch_size, self.hidden_size, device=self.device),
                  torch.zeros(self.bidirectional * self.layers, self.batch_size, self.hidden_size, device=self.device))
        embedded = self.embedding(input).transpose(0, 1)
        output = embedded.view(-1, self.batch_size, self.hidden_size)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.bidirectional * self.layers, self.batch_size, self.hidden_size, device=self.device),\
               torch.zeros(self.bidirectional * self.layers, self.batch_size, self.hidden_size, device=self.device)

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def to(self, device):
        super(EncoderRNN, self).to(device)
        self.device = device
        

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, lang, bidirectional=const.BIDIRECTIONAL, 
                 layers=const.DECODER_LAYERS, dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lang = lang
        self.bidirectional = bidirectional
        self.layers = layers
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size).to(self.device)
        self.lstm = nn.LSTM(hidden_size, hidden_size, self.layers,
                            bidirectional=True if self.bidirectional == 2 else False,
                            dropout=self.dropout).to(self.device)
        self.out = nn.Linear(self.bidirectional * hidden_size, output_size).to(self.device)
        self.softmax = nn.LogSoftmax(dim=1).to(self.device)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, self.batch_size, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[-1]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.layers, self.batch_size, self.hidden_size, device=self.device),\
               torch.zeros(self.layers, self.batch_size, self.hidden_size, device=self.device)

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
    
    def to(self, device):
        super(DecoderRNN, self).to(device)
        self.device = device
