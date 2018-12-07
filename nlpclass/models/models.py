import torch
import torch.nn as nn
import torch.nn.functional as F

from nlpclass.config import model_config


class EncoderRNN(nn.Module):
    def __init__(self, input_size,
                 hidden_size=256, num_layers=1, dropout=0.0, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.grus = nn.ModuleList(
            [nn.GRU(hidden_size, hidden_size,
                    dropout=dropout, bidirectional=bidirectional)])
        self.grus.extend([nn.GRU(hidden_size, hidden_size,
                                 dropout=dropout, bidirectional=bidirectional)
                          for i in range(1, num_layers - 1)])

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i, layer in enumerate(self.grus):
            output, hidden = self.grus[i](output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=model_config.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.grus = nn.ModuleList([nn.GRU(hidden_size, hidden_size)])
        self.grus.extend([nn.GRU(hidden_size, hidden_size)
                          for i in range(1, self.num_layers - 1)])
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        for i, layer in enumerate(self.grus):
            output, hidden = self.grus[i](output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=model_config.device)


class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self):
        pass
