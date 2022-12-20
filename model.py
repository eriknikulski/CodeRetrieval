import random
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import const


class Architecture:
    class Type(Enum):
        DOC = 0
        CODE = 1

    class Mode(Enum):
        NORMAL = 0
        DOC_DOC = 1
        CODE_CODE = 2
        DOC_CODE = 3
        CODE_DOC = 4

    def __init__(self, arch: Mode = Mode.NORMAL):
        assert isinstance(arch, Architecture.Mode)
        self.encoders = []
        self.decoders = []

        if arch == Architecture.Mode.DOC_DOC:
            self.encoders.append(Architecture.Type.DOC)
            self.decoders.append(Architecture.Type.DOC)
        if arch == Architecture.Mode.CODE_CODE:
            self.encoders.append(Architecture.Type.CODE)
            self.decoders.append(Architecture.Type.CODE)
        if arch == Architecture.Mode.DOC_CODE:
            self.encoders.append(Architecture.Type.DOC)
            self.decoders.append(Architecture.Type.CODE)
        if arch == Architecture.Mode.CODE_DOC:
            self.encoders.append(Architecture.Type.CODE)
            self.decoders.append(Architecture.Type.DOC)
        if arch == Architecture.Mode.NORMAL:
            self.encoders.append(Architecture.Type.DOC)
            self.encoders.append(Architecture.Type.CODE)
            self.decoders.append(Architecture.Type.DOC)
            self.decoders.append(Architecture.Type.CODE)

        self.n_encoders = len(self.encoders)
        self.n_decoders = len(self.decoders)

    def get_encoders(self, doc_encoder, code_encoder):
        encoders = []
        if Architecture.Type.DOC in self.encoders:
            encoders.append(doc_encoder)
        if Architecture.Type.CODE in self.encoders:
            encoders.append(code_encoder)
        return encoders

    def get_decoders(self, doc_decoder, code_decoder):
        decoders = []
        if Architecture.Type.DOC in self.decoders:
            decoders.append(doc_decoder)
        if Architecture.Type.CODE in self.decoders:
            decoders.append(code_decoder)
        return decoders

    def get_rand_encoder_id(self):
        if self.n_encoders == 1:
            return 0
        return random.randint(0, 1)

    def get_encoder_input(self, encoder_id, doc_seqs, doc_seq_lengths, code_seqs, code_seq_lengths, methode_names,
                          methode_name_lengths, code_tokens):
        if self.encoders[encoder_id] == Architecture.Type.DOC:
            return doc_seqs, doc_seq_lengths
        else:
            return code_seqs, code_seq_lengths, methode_names, methode_name_lengths, code_tokens

    def get_decoder_sizes(self, doc_size, code_size):
        decoder_sizes = []
        if Architecture.Type.DOC in self.decoders:
            decoder_sizes.append(doc_size)
        if Architecture.Type.CODE in self.decoders:
            decoder_sizes.append(code_size)
        return decoder_sizes


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 layers=const.ENCODER_LAYERS, lstm_dropout=const.LSTM_ENCODER_DROPOUT, device=const.DEVICE):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.layers = layers
        self.lstm_dropout = lstm_dropout
        self.device = device

        self.embedding = nn.Embedding(self.input_size, self.hidden_size, device=self.device)
        self.lstm = nn.LSTM(hidden_size, hidden_size, self.layers, bidirectional=(self.bidirectional == 2),
                            dropout=self.lstm_dropout, batch_first=True, device=self.device)

        if const.GRADIENT_CLIPPING_ENABLED:
            for p in self.lstm.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -const.GRADIENT_CLIPPING_VALUE, const.GRADIENT_CLIPPING_VALUE))

    def forward(self, input, lengths=None):
        hidden = self.init_hidden()
        embedded = self.embedding(input)
        if lengths:
            embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(embedded, hidden)
        if lengths:
            output, _ = pad_packed_sequence(output, batch_first=True, padding_value=const.PAD_TOKEN)
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
    def __init__(self, hidden_size, output_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 layers=const.DECODER_LAYERS, lstm_dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.layers = layers
        self.lstm_dropout = lstm_dropout
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, device=self.device)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, bidirectional=(self.bidirectional == 2),
                            dropout=self.lstm_dropout, batch_first=True, device=self.device)
        self.out = nn.Linear(self.bidirectional * self.hidden_size, self.output_size, device=self.device)
        self.softmax = nn.LogSoftmax(dim=2).to(self.device)

        if const.GRADIENT_CLIPPING_ENABLED:
            for p in self.lstm.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -const.GRADIENT_CLIPPING_VALUE, const.GRADIENT_CLIPPING_VALUE))

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = F.relu(embedded)
        output, hidden = self.lstm(embedded, hidden)
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
    def __init__(self, hidden_size, output_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 layers=const.DECODER_LAYERS, lstm_dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(DecoderRNNWrapped, self).__init__()
        self.decoder = DecoderRNN(hidden_size, output_size, batch_size, bidirectional, layers, lstm_dropout, device)

    def forward(self, encoder_hidden, target_length):
        decoder_input = torch.tensor([[const.SOS_TOKEN]] * self.decoder.batch_size, device=self.decoder.device)
        decoder_hidden = (encoder_hidden,
                          torch.zeros(self.decoder.bidirectional * self.decoder.layers, self.decoder.batch_size,
                                      self.decoder.hidden_size, device=self.decoder.device))

        decoder_outputs = []
        output_seqs = []

        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(dim=1).detach()  # detach from history as input

            decoder_outputs.append(decoder_output)
            output_seqs.append(topi.detach())

        return torch.cat(decoder_outputs, dim=1), torch.cat(output_seqs, dim=1).squeeze()

    def set_batch_size(self, batch_size):
        self.decoder.set_batch_size(batch_size)


class EncoderBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout):
        super(EncoderBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = dropout

    def forward(self, input):
        length = input.size(1)
        max_pool = nn.MaxPool1d(kernel_size=length, stride=length)
        embedded = self.embedding(input)
        out = F.dropout(embedded, self.dropout, self.training)
        out = max_pool(out.transpose(1, 2)).squeeze(2)
        return out


class DocEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 layers=const.ENCODER_LAYERS, lstm_dropout=const.LSTM_ENCODER_DROPOUT, device=const.DEVICE):
        super(DocEncoder, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, batch_size, bidirectional, layers, lstm_dropout, device)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class CodeEncoder(nn.Module):

    def __init__(self, code_vocab_size, hidden_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 lstm_layers=const.ENCODER_LAYERS, lstm_dropout=const.LSTM_ENCODER_DROPOUT, dropout=const.DROPOUT,
                 device=const.DEVICE):
        super(CodeEncoder, self).__init__()
        self.seq_encoder = EncoderRNN(code_vocab_size, hidden_size, batch_size, bidirectional, lstm_layers,
                                      lstm_dropout, device)
        self.methode_name_encoder = EncoderRNN(code_vocab_size, hidden_size, batch_size, bidirectional, lstm_layers,
                                               lstm_dropout, device)
        self.token_encoder = EncoderBOW(code_vocab_size, hidden_size, dropout)

        n = hidden_size
        self.w_seq = nn.Linear(n, n)
        self.w_name = nn.Linear(n, n)
        self.w_tok = nn.Linear(n, n)

        self.fusion = nn.Linear(n, n)

    def forward(self, code_seqs, code_seq_lengths, methode_names, methode_name_length, code_tokens):
        _, (seq_embed, _) = self.seq_encoder(code_seqs, code_seq_lengths)                           # N x L1 x D*H
        _, (name_embed, _) = self.methode_name_encoder(methode_names, methode_name_length)          # N x L2 x D*H
        tok_embed = self.token_encoder(code_tokens)                                                 # N x L3 x D*H

        embed = self.fusion(torch.tanh(self.w_seq(seq_embed) + self.w_name(name_embed) + self.w_tok(tok_embed)))
        return None, (embed, None)


class DocDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 lstm_layers=const.DECODER_LAYERS, lstm_dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(DocDecoder, self).__init__()
        self.decoder = \
            DecoderRNNWrapped(hidden_size, output_size, batch_size, bidirectional, lstm_layers, lstm_dropout, device)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class CodeDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 lstm_layers=const.DECODER_LAYERS, lstm_dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(CodeDecoder, self).__init__()
        self.decoder = \
            DecoderRNNWrapped(hidden_size, output_size, batch_size, bidirectional, lstm_layers, lstm_dropout, device)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class JointEmbedder(nn.Module):
    def __init__(self, arch: Architecture, doc_lang_size, code_lang_size, hidden_size=const.HIDDEN_SIZE,
                 batch_size=const.BATCH_SIZE):
        super(JointEmbedder, self).__init__()

        self.enc_strs = []
        self.dec_strs = []
        self.doc_lang_size = doc_lang_size
        self.code_lang_size = code_lang_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.encoders = nn.ModuleList(arch.get_encoders(
            DocEncoder(self.doc_lang_size, self.hidden_size, self.batch_size),
            CodeEncoder(self.code_lang_size, self.hidden_size, self.batch_size)))
        self.decoders = nn.ModuleList(arch.get_decoders(
            DocDecoder(self.hidden_size, self.doc_lang_size, self.batch_size),
            CodeDecoder(self.hidden_size, self.code_lang_size, self.batch_size)))

    def forward(self, encoder_id, encoder_inputs, decoder_lengths):
        _, encoder_hidden = self.encoders[encoder_id](*encoder_inputs)

        decoder_outputs, output_seqs = [], []

        for i, decoder in enumerate(self.decoders):
            _decoder_outputs, _output_seqs = decoder(encoder_hidden[0], decoder_lengths[i])
            decoder_outputs.append(_decoder_outputs)
            output_seqs.append(_output_seqs)

        return decoder_outputs, output_seqs
