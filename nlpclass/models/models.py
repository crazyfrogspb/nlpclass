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

    def forward(self, x, lengths, hidden=None):
        embed = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embed, lengths, batch_first=True)
        encoded_input, hidden = self.rnn(packed, hidden)
        encoded_input, _ = torch.nn.utils.rnn.pad_packed_sequence(
            encoded_input, padding_value=model_config.PAD_token, batch_first=True)

        if self.bidirectional:
            hidden = torch.cat(
                (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)), 2)

        return encoded_input, hidden


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
            output = self.out(output.squeeze(1))
            return output, hidden, context, encoder_output


def calc_loss(logits, target, criterion):
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target.view(-1, 1).squeeze()
    return criterion(logits_flat, target_flat)


class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder,
                 teacher_forcing_ratio=1.0, beam_size=5, beam_alpha=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = model_config.max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_size = beam_size
        self.beam_alpha = beam_alpha

    def encode_sentence(self, input_seq, input_length):
        batch_size = input_seq.size(0)

        encoded_input, encoder_hidden = self.encoder(input_seq, input_length)

        decoder_hidden = encoder_hidden[-1].unsqueeze(0)
        context = None
        if self.decoder.attention:
            context = Variable(torch.zeros(encoded_input.size(
                0), encoded_input.size(2))).unsqueeze(1).to(model_config.device)

        decoder_input = Variable(torch.LongTensor(
            [model_config.SOS_token] * batch_size)).to(model_config.device)

        return encoded_input, decoder_hidden, decoder_input, context

    def forward(self, x):
        input_seq = x['input']
        target_seq = x['target']
        input_length = x['input_length']
        target_length = x['target_length']
        batch_size = input_seq.size(0)

        encoder_output, decoder_hidden, decoder_input, context = self.encode_sentence(
            input_seq, input_length)

        if random.random() < self.teacher_forcing_ratio:
            use_teacher_forcing = True
        else:
            use_teacher_forcing = False

        decoder_outputs = Variable(torch.zeros(
            model_config.max_length + 1, batch_size, self.decoder.output_size)).to(model_config.device)

        for idx in range(target_seq.size(1) - 1):
            decoder_output, decoder_hidden, context, weights = self.decoder(
                decoder_input, decoder_hidden, encoder_output, context)

            decoder_outputs[idx] = decoder_output

            if use_teacher_forcing:
                decoder_input = target_seq[:, idx]
            else:
                _, topi = decoder_output.data.topk(1)
                decoder_input = Variable(topi).squeeze()
                decoder_input = decoder_input.to(model_config.device)

            decoder_hidden.detach()

        return decoder_outputs[:target_length.max()].transpose(0, 1).contiguous()

    def greedy(self, x):
        input_seq = x['input']
        input_length = x['input_length']
        batch_size = input_seq.size(0)

        encoder_output, decoder_hidden, decoder_input, context = self.encode_sentence(
            input_seq, input_length)

        predictions = torch.LongTensor(
            [model_config.SOS_token] * batch_size).unsqueeze(1).to(model_config.device)

        for input_idx in range(self.max_length):
            decoder_output, decoder_hidden, context, weights = self.decoder(
                decoder_input, decoder_hidden, encoder_output, context)

            topv, topi = decoder_output.topk(1)
            predictions = torch.cat((predictions, topi), dim=1)

            decoder_input = Variable(topi).squeeze().to(model_config.device)

            num_done = ((predictions == model_config.EOS_token).sum(
                dim=1) > 0).sum().cpu().numpy()
            if num_done == batch_size:
                return predictions

        return predictions

    def beam(self, decoder_input, decoder_hidden, encoder_output, context):
        indices = []
        probs = []

        decoder_output, decoder_hidden, context, weights = self.decoder(
            decoder_input, decoder_hidden, encoder_output, context)

        topv, topi = decoder_output.topk(self.beam_size)

        for x in range(self.beam_size):
            indices.append(topi[0][x].item())
            probs.append(topv[0][x].item())

        return indices, probs, decoder_hidden, context

    def beam_init(self, decoder_input, decoder_hidden, encoder_output, context):
        beams_keep = [[] for i in range(self.beam_size)]

        decoder_output, decoder_hidden, context, weights = self.decoder(
            decoder_input, decoder_hidden, encoder_output, context)
        topv, topi = decoder_output.topk(self.beam_size)

        for x in range(self.beam_size):
            beams_keep[x].append({'decoder_hidden': decoder_hidden,
                                  'decoder_input': topi[:, x],
                                  'loss': topv[:, x],
                                  'context': context})

        return beams_keep

    def beam_step(self, beams_keep, step):
        beams = [[] for i in range(self.beam_size ** 2)]
        probs_vec = np.zeros(self.beam_size ** 2)

        for beam_num, beam in enumerate(beams_keep):
            decoder_output, decoder_hidden, context, weights = self.decoder(
                beam['decoder_input'], beam['decoder_hidden'],
                encoder_output, beam['context'])

    def beam_search(self, x):
        input_seq = x['input']
        input_length = x['input_length']
        batch_size = input_seq.size(0)

        encoder_output, decoder_hidden, decoder_input, context = self.encode_sentence(
            input_seq, input_length)

        for idx in range(1, self.max_length):
            beams, probs = self.beam_step(beams_keep, idx)
            best_inds = np.argpartition(
                probs, -self.beam_size)[-self.beam_size:]
            diff = self.beam_size - len(beams_keep)
            if diff > 0:
                for beam in range(diff):
                    beams_keep.append([])
            for ind_num, ind in enumerate(best_inds):
                beams_keep[ind_num] = beams[ind]

            beam_copy = beams_keep.copy()
            for beam in beam_copy:
                all_words = [beam_node['decoder_input'] for beam_node in beam]
                step_len = len(all_words)
                if model_config.EOS_token in all_words:
                    associated_prob = np.prod(
                        np.array([beam_node['loss'] for beam_node in beam])) / step_len**self.beam_alpha
                    final_sentences.append((all_words, associated_prob))
                    beams_keep.remove(x)
            if len(final_sentences) > self.beam_size:
                final_probs = np.array([sentence[1]
                                        for sentence in final_sentences])
                return final_sentences[final_probs.argmax()][0]
