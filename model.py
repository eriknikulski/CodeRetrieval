import torch
import torch.nn as nn
import torch.nn.functional as F

import const


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, lang, bidirectional=const.BIDIRECTIONAL, 
                 layers=const.ENCODER_LAYERS, dropout=const.LSTM_ENCODER_DROPOUT, device=const.DEVICE):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lang = lang
        self.bidirectional = bidirectional
        self.layers = layers
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(self.input_size, self.hidden_size, device=self.device)
        self.lstm = nn.LSTM(hidden_size, hidden_size, self.layers, bidirectional=(self.bidirectional == 2), 
                            dropout=self.dropout, batch_first=True, device=self.device)

    def forward(self, input):
        hidden = self.init_hidden()
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.bidirectional * self.layers, self.batch_size, self.hidden_size, device=self.device),\
               torch.zeros(self.bidirectional * self.layers, self.batch_size, self.hidden_size, device=self.device)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def to(self, device):
        self.device = device
        return super(EncoderRNN, self).to(device)
        

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, lang, bidirectional=const.BIDIRECTIONAL, 
                 layers=const.DECODER_LAYERS, dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.lang = lang
        self.bidirectional = bidirectional
        self.layers = layers
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, device=self.device)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, bidirectional=(self.bidirectional == 2), 
                            dropout=self.dropout, batch_first=True, device=self.device)
        self.out = nn.Linear(self.bidirectional * self.hidden_size, self.output_size, device=self.device)
        self.softmax = nn.LogSoftmax(dim=2).to(self.device)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.bidirectional * self.layers, self.batch_size, self.hidden_size, device=self.device),\
               torch.zeros(self.bidirectional * self.layers, self.batch_size, self.hidden_size, device=self.device)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def to(self, device):
        self.device = device
        return super(DecoderRNN, self).to(device)


class DecoderRNNWrapped(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, lang, bidirectional=const.BIDIRECTIONAL, 
                 layers=const.DECODER_LAYERS, dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(DecoderRNNWrapped, self).__init__()
        self.decoder = DecoderRNN(hidden_size, output_size, batch_size, lang, bidirectional, layers, dropout, device)

    def forward(self, encoder_hidden, target_length):
        decoder_input = torch.tensor([[const.SOS_TOKEN]] * self.decoder.batch_size, device=self.decoder.device)
        decoder_hidden = (encoder_hidden,
                          torch.zeros(self.decoder.bidirectional * self.decoder.layers, self.decoder.batch_size, 
                                      self.decoder.hidden_size, device=self.decoder.device))

        decoder_outputs = []
        output_seqs = []

        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(dim=1).detach()  # detach from history as input

            decoder_outputs.append(decoder_output)
            output_seqs.append(topi.detach())

        return torch.cat(decoder_outputs, dim=1), torch.cat(output_seqs, dim=1).squeeze()

    def set_batch_size(self, batch_size):
        self.decoder.set_batch_size(batch_size)
