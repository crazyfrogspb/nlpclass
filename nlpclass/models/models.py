import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nlpclass.config import model_config


class EncoderRNN(nn.Module):
    def __init__(self, input_size,
                 embedding_size=100, hidden_size=64, num_layers=1,
                 dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers,
                          batch_first=True, bidirectional=self.bidirectional,
                          dropout=self.dropout)

    def forward(self, x, lengths):
        batch_size, seq_len = x.size()

        hidden = self.init_hidden(batch_size)
        embed = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embed, lengths, batch_first=True)
        output, hidden = self.rnn(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)

        if self.bidirectional:
            hidden = torch.cat(
                (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)), 2)

        return output, hidden

    def init_hidden(self, batch_size):
        if self.bidirectional:
            multiplier = 2
        else:
            multiplier = 1
        hidden = torch.randn(multiplier * self.num_layers, batch_size,
                             self.hidden_size, device=model_config.device)
        return hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden, encoder_output):
        energies = self.attn(
            torch.cat((hidden.expand(*encoder_output.size()), encoder_output), 2))
        energies = torch.bmm(energies, self.v.unsqueeze(
            0).expand(*hidden.size()).transpose(1, 2)).squeeze(2)

        return F.softmax(energies, 1)


class DecoderRNN(nn.Module):
    def __init__(self, output_size,
                 embedding_size=100, hidden_size=64,
                 num_layers=1, attention=False):
        super().__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention

        if self.attention:
            self.attention_layer = Attention(self.hidden_size)
            rnn_input_size = self.embedding_size + self.hidden_size
            self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        else:
            rnn_input_size = self.embedding_size
            self.out = nn.Linear(self.hidden_size, self.output_size)

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.rnn = nn.GRU(rnn_input_size, self.hidden_size,
                          self.num_layers, batch_first=True)

    def forward(self, input, hidden, encoder_output=None, context=None):
        embed = self.embedding(input).unsqueeze(1)

        if self.attention:
            embed = torch.cat((embed, context), 2)
            output, hidden = self.rnn(embed, hidden)
            weights = self.attention_layer(output, encoder_output)
            context = weights.unsqueeze(1).bmm(encoder_output)
            output = F.log_softmax(
                self.out(torch.cat((context.squeeze(1), output.squeeze(1)), 1)), 1)
            return output, hidden, context, weights
        else:
            output, hidden = self.rnn(embed, hidden)
            output = F.log_softmax(self.out(output.squeeze(1)), 1)
            return output, hidden, context, encoder_output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=model_config.device)


def calculate_loss(decoder_output, input_index, input_tokens, input_length):
    mask = torch.LongTensor(np.repeat([input_index], decoder_output.size(0)))
    mask = Variable(mask).to(model_config.device)
    mask = mask < input_length

    return -torch.gather(decoder_output, dim=1,
                         index=input_tokens.unsqueeze(1)).squeeze() * mask.float()


class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder, max_length=100, teacher_forcing_ratio=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, x):
        input_seq = x['input']
        target_seq = x['target']
        input_length = x['input_length']
        batch_size = input_seq.size(0)

        encoder_output, encoder_hidden = self.encoder(input_seq, input_length)

        decoder_hidden = encoder_hidden
        context = None
        if self.decoder.attention:
            context = Variable(torch.zeros(encoder_output.size(
                0), encoder_output.size(2))).unsqueeze(1).to(model_config.device)

        total_loss = 0

        if random.random() < self.teacher_forcing_ratio:
            use_teacher_forcing = True
        else:
            use_teacher_forcing = False

        decoder_input = Variable(torch.LongTensor(
            [model_config.SOS_token] * batch_size)).to(model_config.device)

        for input_idx in range(target_seq.size(1) - 1):
            decoder_output, decoder_hidden, context, weights = self.decoder(
                decoder_input, decoder_hidden, encoder_output, context)

            loss = calculate_loss(decoder_output, input_idx,
                                  target_seq[:, input_idx], input_length)
            loss_sum = loss.sum()
            if loss_sum > 0:
                total_loss += loss_sum / torch.sum(loss > 0).float()

            if use_teacher_forcing:
                decoder_input = target_seq[:, input_idx]
            else:
                _, topi = decoder_output.data.topk(1)
                decoder_input = Variable(torch.cat(topi))
                decoder_input = decoder_input.to(model_config.device)

        return total_loss, decoder_output, decoder_hidden

    def greedy(self, x):
        input_seq = x['input']
        input_length = x['input_length']
        batch_size = input_seq.size(0)

        encoder_output, encoder_hidden = self.encoder(input_seq, input_length)

        decoder_hidden = encoder_hidden
        context = None
        if self.decoder.attention:
            context = Variable(torch.zeros(encoder_output.size(
                0), encoder_output.size(2))).unsqueeze(1).to(model_config.device)

        decoder_input = Variable(torch.LongTensor(
            [model_config.SOS_token] * batch_size)).to(model_config.device)
        predictions = torch.LongTensor(
            [model_config.SOS_token] * batch_size).unsqueeze(1).to(model_config.device)

        for input_idx in range(self.max_length):
            decoder_output, decoder_hidden, context, weights = self.decoder(
                decoder_input, decoder_hidden, encoder_output, context)

            topv, topi = decoder_output.topk(1)
            predictions = torch.cat((predictions, topi), dim=1)

            num_done = ((predictions == model_config.EOS_token).sum(
                dim=1) > 0).sum().cpu().numpy()
            if num_done == batch_size:
                return predictions

        return predictions

    def beam(self, x):
        input_seq = x['input']
        input_length = x['input_length']
        batch_size = input_seq.size(0)

        encoder_output, encoder_hidden = self.encoder(input_seq, input_length)

        decoder_hidden = encoder_hidden
        context = None
        if self.decoder.attention:
            context = Variable(torch.zeros(encoder_output.size(
                0), encoder_output.size(2))).unsqueeze(1).to(model_config.device)
