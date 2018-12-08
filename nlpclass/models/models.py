import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        input_seq = x['input']
        target_seq = x['target']
        input_length = x['input_length']

        encoder_output, encoder_hidden = self.encoder(input_seq, input_length)
