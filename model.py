import random
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import const

# TODO: set padding_token, max_norm in nn.Embedding


class ModelType(Enum):
    TRANSLATOR = 0
    EMBEDDER = 1


class Architecture:
    class Type(Enum):
        DOC = 0
        CODE = 1

    class Mode(Enum):
        NORMAL = 'normal'
        DOC_DOC = 'doc_doc'
        CODE_CODE = 'code_code'
        DOC_CODE = 'doc_code'
        CODE_DOC = 'code_doc'

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

    def get_encoder_elements(self, encoder_id, doc_elem, code_elem):
        if self.encoders[encoder_id] == Architecture.Type.DOC:
            return doc_elem
        return code_elem

    def get_encoder_input(self, encoder_id, doc_inputs, code_inputs):
        return self.get_encoder_elements(encoder_id, doc_inputs, code_inputs)

    def get_encoder_lang(self, encoder_id, doc_lang, code_lang):
        return self.get_encoder_elements(encoder_id, doc_lang, code_lang)

    def get_decoder_elements(self, doc_elem, code_elem):
        out = []
        if Architecture.Type.DOC in self.decoders:
            out.append(doc_elem)
        if Architecture.Type.CODE in self.decoders:
            out.append(code_elem)
        return out

    def get_decoder_sizes(self, doc_size, code_size):
        return self.get_decoder_elements(doc_size, code_size)

    def get_decoder_languages(self, doc_lang, code_lang):
        return self.get_decoder_elements(doc_lang, code_lang)

    def get_decoder_targets(self, doc_target, code_target):
        return self.get_decoder_elements(doc_target, code_target)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, embedding, bidirectional=const.BIDIRECTIONAL,
                 layers=const.ENCODER_LAYERS, lstm_dropout=const.LSTM_ENCODER_DROPOUT, device=const.DEVICE):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.layers = layers
        self.lstm_dropout = lstm_dropout
        self.device = device

        self.embedding = embedding

        self.lstm = nn.LSTM(hidden_size, hidden_size, self.layers, bidirectional=(self.bidirectional == 2),
                            dropout=self.lstm_dropout, batch_first=True, device=self.device)

    def forward(self, input, lengths=None):
        hidden = self.init_hidden()
        embedded = self.embedding(input)
        if lengths is not None:
            embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(embedded, hidden)
        if lengths is not None:
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
    def __init__(self, hidden_size, output_size, batch_size, embedding, bidirectional=const.BIDIRECTIONAL,
                 layers=const.DECODER_LAYERS, lstm_dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.layers = layers
        self.lstm_dropout = lstm_dropout
        self.device = device

        self.embedding = embedding

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, bidirectional=(self.bidirectional == 2),
                            dropout=self.lstm_dropout, batch_first=True, device=self.device)
        self.out = nn.Linear(self.bidirectional * self.hidden_size, self.output_size, device=self.device)
        self.softmax = nn.LogSoftmax(dim=2).to(self.device)

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
    def __init__(self, hidden_size, output_size, batch_size, embedding, bidirectional=const.BIDIRECTIONAL,
                 layers=const.DECODER_LAYERS, lstm_dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(DecoderRNNWrapped, self).__init__()
        self.decoder = DecoderRNN(hidden_size, output_size, batch_size, embedding, bidirectional, layers, lstm_dropout,
                                  device)

    def forward(self, encoder_hidden, target_length):
        decoder_input = torch.tensor([[const.SOS_TOKEN]] * self.decoder.batch_size, device=self.decoder.device)
        decoder_hidden = (encoder_hidden,
                          torch.zeros(self.decoder.bidirectional * self.decoder.layers, self.decoder.batch_size,
                                      self.decoder.hidden_size, device=self.decoder.device))

        decoder_outputs = []

        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(dim=1).detach()  # detach from history as input

            decoder_outputs.append(decoder_output)
        return torch.cat(decoder_outputs, dim=1), decoder_hidden

    def set_batch_size(self, batch_size):
        self.decoder.set_batch_size(batch_size)


class EncoderBOW(nn.Module):
    def __init__(self, embedding, dropout):
        super(EncoderBOW, self).__init__()
        self.embedding = embedding
        self.dropout = dropout

    def forward(self, input):
        length = input.size(1)
        embedded = self.embedding(input)
        out = F.dropout(embedded, self.dropout, self.training)
        out = F.max_pool1d(out.transpose(1, 2), kernel_size=length, stride=length).squeeze(2)
        return out


class DocEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 layers=const.ENCODER_LAYERS, lstm_dropout=const.LSTM_ENCODER_DROPOUT, device=const.DEVICE):
        super(DocEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=const.PAD_TOKEN, device=device)
        self.encoder = EncoderRNN(input_size, hidden_size, batch_size, self.embedding,
                                  bidirectional, layers, lstm_dropout, device)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class CodeEncoder(nn.Module):

    def __init__(self, code_vocab_size, hidden_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 lstm_layers=const.ENCODER_LAYERS, lstm_dropout=const.LSTM_ENCODER_DROPOUT, dropout=const.DROPOUT,
                 device=const.DEVICE, simple=False):
        super(CodeEncoder, self).__init__()
        self.simple = simple
        self.embedding = nn.Embedding(code_vocab_size, hidden_size, padding_idx=const.PAD_TOKEN, device=device)
        self.seq_encoder = EncoderRNN(code_vocab_size, hidden_size, batch_size, self.embedding,
                                      bidirectional, lstm_layers, lstm_dropout, device)
        if not self.simple:
            self.methode_name_encoder = EncoderRNN(code_vocab_size, hidden_size, batch_size, self.embedding,
                                                   bidirectional, lstm_layers, lstm_dropout, device)
            self.token_encoder = EncoderBOW(self.embedding, dropout)

            n = hidden_size
            self.w_seq = nn.Linear(n, n)
            self.w_name = nn.Linear(n, n)
            self.w_tok = nn.Linear(n, n)

            self.fusion = nn.Linear(n, n)

    def forward(self, code_seqs, code_seq_lengths, methode_names, methode_name_length, code_tokens, code_tokens_length):
        _, (seq_embed, _) = self.seq_encoder(code_seqs, code_seq_lengths)                           # N x L1 x D*H
        if self.simple:
            return None, (seq_embed, None)
        _, (name_embed, _) = self.methode_name_encoder(methode_names, methode_name_length)          # N x L2 x D*H
        tok_embed = self.token_encoder(code_tokens)                                                 # N x L3 x D*H

        embed = self.fusion(torch.tanh(self.w_seq(seq_embed) + self.w_name(name_embed) + self.w_tok(tok_embed)))
        return None, (embed, None)


class DocDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 lstm_layers=const.DECODER_LAYERS, lstm_dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(DocDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=const.PAD_TOKEN, device=device)
        self.decoder = \
            DecoderRNNWrapped(hidden_size, output_size, batch_size, self.embedding,
                              bidirectional, lstm_layers, lstm_dropout, device)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class CodeDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, bidirectional=const.BIDIRECTIONAL,
                 lstm_layers=const.DECODER_LAYERS, lstm_dropout=const.LSTM_DECODER_DROPOUT, device=const.DEVICE):
        super(CodeDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=const.PAD_TOKEN, device=device)
        self.decoder = \
            DecoderRNNWrapped(hidden_size, output_size, batch_size, self.embedding,
                              bidirectional, lstm_layers, lstm_dropout, device)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class JointTranslator(nn.Module):
    def __init__(self, arch: Architecture, doc_lang_size, code_lang_size, hidden_size=const.HIDDEN_SIZE,
                 batch_size=const.BATCH_SIZE, simple=False):
        super(JointTranslator, self).__init__()

        self.enc_strs = []
        self.dec_strs = []
        self.doc_lang_size = doc_lang_size
        self.code_lang_size = code_lang_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.encoders = nn.ModuleList(arch.get_encoders(
            DocEncoder(self.doc_lang_size, self.hidden_size, self.batch_size),
            CodeEncoder(self.code_lang_size, self.hidden_size, self.batch_size, simple=simple)))
        self.decoders = nn.ModuleList(arch.get_decoders(
            DocDecoder(self.hidden_size, self.doc_lang_size, self.batch_size),
            CodeDecoder(self.hidden_size, self.code_lang_size, self.batch_size)))

    def forward(self, encoder_id, encoder_inputs, decoder_lengths):
        _, encoder_hidden = self.encoders[encoder_id](*encoder_inputs)

        decoder_outputs, output_seqs = [], []

        for i, decoder in enumerate(self.decoders):
            _decoder_outputs, _ = decoder(encoder_hidden[0], decoder_lengths[i])
            _output_seqs = _decoder_outputs.topk(1)[1].squeeze()
            decoder_outputs.append(_decoder_outputs)
            output_seqs.append(_output_seqs)

        return decoder_outputs, output_seqs


class JointEmbedder(nn.Module):
    def __init__(self, doc_lang_size, code_lang_size, hidden_size=const.HIDDEN_SIZE, batch_size=const.BATCH_SIZE,
                 simple=False):
        super(JointEmbedder, self).__init__()

        self.enc_strs = []
        self.dec_strs = []
        self.doc_lang_size = doc_lang_size
        self.code_lang_size = code_lang_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.doc_encoder = DocEncoder(self.doc_lang_size, self.hidden_size, self.batch_size)
        self.code_encoder = CodeEncoder(self.code_lang_size, self.hidden_size, self.batch_size, simple=simple)

    def forward(self, doc_inputs, code_inputs, neg_doc_inputs, neg_code_inputs):
        _, (doc_hidden, _) = self.doc_encoder(*doc_inputs)
        _, (code_hidden, _) = self.code_encoder(*code_inputs)

        _, (neg_doc_hidden, _) = self.doc_encoder(*neg_doc_inputs)
        _, (neg_code_hidden, _) = self.code_encoder(*neg_code_inputs)

        # TODO: richtige variations?? In paper nur 0, 1
        # TODO: in deep_code_search margin = 0.05 ???
        variations = [
            (doc_hidden, code_hidden, torch.tensor(1)),
            (neg_doc_hidden, code_hidden, torch.tensor(-1)),
            # (doc_hidden, neg_code_hidden, torch.tensor(-1)),
        ]

        # TODO: reduction in paper ist sum ?? -> bei 2 * -1  --> macht sum sinn?
        margin = 0.05
        return (margin - sum(var[2] * F.cosine_similarity(var[0], var[1]) for var in variations)).clamp(min=1e-6).mean()
        # TODO: use cosine embedding??
        # return sum(F.cosine_embedding_loss(*var) for var in variations)
