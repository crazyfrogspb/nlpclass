"""
Definitions of all model classes
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nlpclass.config import model_config


class EncoderCNN(nn.Module):
    def __init__(self, input_size, num_layers=2,
                 embedding_size=100, hidden_size=64, kernel_size=7):
        """Convolutional encoder class.

        Parameters
        ----------
        input_size : int
            Size of the vocabulary.
        num_layers : int, optional
            Number of convolutional layers (the default is 2).
        embedding_size : int, optional
            Embedding size (the default is 100).
        hidden_size : int, optional
            Size of the convolutional layers (the default is 64).
        kernel_size : int
            Kernel size (the default is 7).
        """
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            self.input_size, self.embedding_size, padding_idx=model_config.PAD_token)

        self.conv_encoder = nn.ModuleList([nn.Conv1d(
            self.embedding_size, self.hidden_size, kernel_size=self.kernel_size, padding=self.padding)])
        for i in range(1, self.num_layers):
            self.conv_encoder.append(nn.Conv1d(
                self.hidden_size, self.hidden_size, kernel_size=self.kernel_size, padding=self.padding))

    def forward(self, x, lengths):
        batch_size, seq_len = x.size()
        hidden = self.embedding(x)

        for layer in self.conv_encoder:
            hidden = layer(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(
                batch_size, seq_len, hidden.size(-1))

        hidden = torch.max(hidden, dim=1)[0]

        return None, hidden


class EncoderRNN(nn.Module):
    def __init__(self, input_size,
                 embedding_size=100, hidden_size=64, num_layers=1,
                 dropout=0.0, bidirectional=False, pretrained_embeddings=None):
        """Recurrent encoder class.

        Parameters
        ----------
        input_size : int
            Size of the input vocabulary.
        embedding_size : int, optional
            Embedding size (the default is 100).
        hidden_size : int, optional
            Size of the recurrent layer (the default is 64).
        num_layers : int, optional
            Number of layers (the default is 1).
        dropout : float, optional
            Dropout between recurrent layers (the default is 0.0).
        bidirectional : bool, optional
            Whether to make recurrent layer bidirectional (the default is False).
        pretrained_embeddings : np.array, optional
            Array of pretrained word embeddings (the default is None).
        """
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            self.input_size, self.embedding_size, padding_idx=model_config.PAD_token)
        if pretrained_embeddings is not None:
            self.init_weights(pretrained_embeddings)
        else:
            self.embedding.weight.data.uniform_(
                -model_config.embedding_init, model_config.embedding_init)

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers,
                          batch_first=True, bidirectional=self.bidirectional,
                          dropout=self.dropout)

    def forward(self, x, lengths, hidden=None):
        embed = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embed, lengths, batch_first=True)
        encoder_output, hidden = self.rnn(packed, hidden)
        encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            encoder_output, batch_first=True)

        if self.bidirectional:
            hidden = torch.cat(
                (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)), 2)

        return encoder_output, hidden

    def init_weights(self, pretrained_embeddings):
        if pretrained_embeddings is not None:
            self.embedding.weight.data = torch.from_numpy(
                pretrained_embeddings).float()


class Attention(nn.Module):
    def __init__(self, hidden_size):
        """Luong's general attention layer class.

        Parameters
        ----------
        hidden_size : int
            Size of the attention layer.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_output):
        encoder_output = encoder_output.contiguous()
        scores = self.attn(encoder_output.view(-1, self.hidden_size))
        scores = scores.view(*encoder_output.size())
        scores = torch.bmm(scores, hidden.transpose(1, 2)).squeeze(2)

        return F.softmax(scores, dim=1)


class DecoderRNN(nn.Module):
    def __init__(self, output_size,
                 embedding_size=100, hidden_size=64,
                 num_layers=1, attention=False, pretrained_embeddings=None):
        """Recurrent decoder class.

        Parameters
        ----------
        output_size : int
            Size of the target vocabulary.
        embedding_size : int, optional
            Embedding size (the default is 100).
        hidden_size : int, optional
            Size of the recurrent layer (the default is 64).
        num_layers : int, optional
            Number of layers (the default is 1).
        attention : boolean, optional
            Whether to use attention mechanism (the default is False).
        pretrained_embeddings : np.array, optional
            Array of pretrained word embeddings (the default is None).
        """
        super().__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention

        rnn_input_size = self.embedding_size + self.hidden_size

        if self.attention:
            self.attention_layer = Attention(self.hidden_size)
            self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
            self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        else:
            self.out = nn.Linear(self.hidden_size, self.output_size)

        self.embedding = nn.Embedding(
            self.output_size, self.embedding_size, padding_idx=model_config.PAD_token)
        if pretrained_embeddings is not None:
            self.init_weights(pretrained_embeddings)
        else:
            self.embedding.weight.data.uniform_(
                -model_config.embedding_init, model_config.embedding_init)

        self.rnn = nn.GRU(rnn_input_size, self.hidden_size,
                          self.num_layers, batch_first=True)

    def forward(self, input, hidden, encoder_output=None, context=None):
        embed = self.embedding(input).unsqueeze(1)
        embed = torch.cat((embed, context), 2)
        output, hidden = self.rnn(embed, hidden)

        if self.attention:
            weights = self.attention_layer(output, encoder_output)
            weights = weights.unsqueeze(1)
            context = weights.bmm(encoder_output)
            hidden_context = torch.cat(
                (context.squeeze(1), hidden.squeeze(0)), dim=1)
            hidden_context = torch.tanh(self.concat(hidden_context))
            output = self.out(hidden_context)
            output = F.log_softmax(output, dim=1)
            return output, hidden, context, weights
        else:
            output = self.out(output.squeeze(1))
            output = F.log_softmax(output, dim=1)
            return output, hidden, context, None

    def init_weights(self, pretrained_embeddings):
        if pretrained_embeddings is not None:
            self.embedding.weight.data = torch.from_numpy(
                pretrained_embeddings).float()


def calculate_loss(decoder_output, idx, target_tokens, target_length):
    # calculate masked loss for the given index
    mask = torch.LongTensor(np.repeat([idx], decoder_output.size(0)))
    mask = Variable(mask).to(model_config.device)
    mask = mask < target_length
    loss = -decoder_output.gather(1, target_tokens.view(-1, 1))
    masked_loss = loss * mask.unsqueeze(1).float()
    return masked_loss.squeeze(1)


class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder,
                 teacher_forcing_ratio=1.0, beam_size=5, beam_alpha=1.0):
        """Wrapper for Seq2Seq model.

        Parameters
        ----------
        encoder : EncoderCNN or EncoderRNN
            Encoder instance.
        decoder : DecoderRNN
            Decoder instance.
        teacher_forcing_ratio : float, optional
            Teacher forcing ratio (the default is 1.0).
        beam_size : int, optional
            Size of the beam (the default is 5).
        beam_alpha : float, optional
            Length penalty (the default is 1.0).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = model_config.max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_size = beam_size
        self.beam_alpha = beam_alpha

    def encode_sentence(self, input_seq, input_length):
        batch_size = input_seq.size(0)

        encoder_output, encoder_hidden = self.encoder(input_seq, input_length)

        if len(encoder_hidden.size()) == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)

        # get the hidden state of the last recurrent layer
        decoder_hidden = encoder_hidden[-1].unsqueeze(0)

        if self.decoder.attention:
            context = Variable(torch.zeros(encoder_output.size(
                0), encoder_output.size(2))).unsqueeze(1).to(model_config.device)
        else:
            context = decoder_hidden.transpose(0, 1)

        decoder_input = Variable(torch.LongTensor(
            [model_config.SOS_token] * batch_size)).to(model_config.device)

        return encoder_output, decoder_hidden, decoder_input, context

    def forward(self, x):
        input_seq = x['input']
        target_seq = x['target']
        input_length = x['input_length']
        target_length = x['target_length']

        encoder_output, decoder_hidden, decoder_input, context = self.encode_sentence(
            input_seq, input_length)

        if random.random() < self.teacher_forcing_ratio:
            use_teacher_forcing = True
        else:
            use_teacher_forcing = False

        total_loss = 0

        for idx in range(target_seq.size(1)):
            decoder_output, decoder_hidden, context, weights = self.decoder(
                decoder_input, decoder_hidden, encoder_output, context)

            loss = calculate_loss(decoder_output, idx,
                                  target_seq[:, idx], target_length)
            total_loss += loss

            if use_teacher_forcing:
                decoder_input = target_seq[:, idx]
            else:
                _, topi = decoder_output.data.topk(1)
                decoder_input = Variable(topi).squeeze(
                    1).to(model_config.device)

            decoder_hidden = decoder_hidden.detach()

        total_loss /= target_length.float()

        return total_loss.mean()

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

            decoder_input = Variable(topi).squeeze(1).to(model_config.device)

        return predictions

    def beam(self, decoder_input, decoder_hidden, encoder_output, context):
        data = []
        decoder_input = decoder_input.unsqueeze(dim=0)
        decoder_output, decoder_hidden, context, weights = self.decoder(
            decoder_input, decoder_hidden, encoder_output, context)
        topv, topi = decoder_output.topk(self.beam_size)

        for x in range(self.beam_size):

            data.append({'decoder_hidden': decoder_hidden,
                         'decoder_input': topi[0, x],
                         'decoder_input_num': topi[0, x].item(),
                         'value': topv[0, x],
                         'value_num': topv[0, x].item(),
                         'context': context})

        return data

    def beams_init(self, decoder_input, decoder_hidden, encoder_output, context):
        beams_keep = [[] for i in range(self.beam_size)]
        decoder_output, decoder_hidden, context, weights = self.decoder(
            decoder_input, decoder_hidden, encoder_output, context)
        topv, topi = decoder_output.topk(self.beam_size)

        for x in range(self.beam_size):
            beams_keep[x].append({'decoder_hidden': decoder_hidden,
                                  'decoder_input': topi[0, x],
                                  'decoder_input_num': topi[0, x].item(),
                                  'value': topv[0, x],
                                  'value_num': topv[0, x].item(),
                                  'context': context})

        return beams_keep

    def beam_step(self, beams_keep, step, encoder_output):
        beams = [[] for i in range(self.beam_size ** 2)]
        probs_vec = np.zeros(self.beam_size ** 2)

        for beam_num, beam_it in enumerate(beams_keep):
            data = \
                self.beam(beam_it[-1]['decoder_input'], beam_it[-1]
                          ['decoder_hidden'], encoder_output, beam_it[-1]['context'])

            beam_prob = 1
            for z in beam_it:
                beam_prob = beam_prob * np.exp(z['value_num'])

            for data_num, sub_beam in enumerate(data):
                # getting the probability for the beams being explored
                probs_vec[data_num + beam_num * self.beam_size] = beam_prob * \
                    np.exp(sub_beam['value_num']) / step**self.beam_alpha
                beams[data_num + beam_num * self.beam_size] = beam_it.copy()
                beams[data_num + beam_num * self.beam_size].append(sub_beam)

        return beams, probs_vec

    def ind_beam(self, beams_keep, encoder_output):
        final_sentences = []
        for idx in range(1, self.max_length):

            beams, prob_vec = self.beam_step(beams_keep, idx, encoder_output)

            # finding the best beams
            best_inds = np.argpartition(
                prob_vec, -self.beam_size)[-self.beam_size:]

            # this is done just in case beams keep is shortened
            diff = self.beam_size - len(beams_keep)
            if diff > 0:
                for beam in range(diff):
                    beams_keep.append([])

            for ind_num, ind in enumerate(best_inds):
                beams_keep[ind_num] = beams[ind]

            beam_copy = beams_keep.copy()

            for beam in beam_copy:
                all_words = [beam_node['decoder_input_num']
                             for beam_node in beam]

                step_len = len(all_words)
                if model_config.EOS_token in all_words or idx == self.max_length - 1:
                    associated_prob = np.prod(
                        np.array([np.exp(beam_node['value_num']) for beam_node in beam])) / step_len**self.beam_alpha
                    final_sentences.append((all_words, associated_prob))
                    beams_keep.remove(beam)

            if len(final_sentences) >= self.beam_size:

                final_probs = np.array([sentence[1]
                                        for sentence in final_sentences])
                return final_sentences[final_probs.argmax()][0]

    def beam_search(self, x):
        input_seq = x['input']
        input_length = x['input_length']
        batch_size = input_seq.size(0)

        encoder_output, decoder_hidden, decoder_input, context = self.encode_sentence(
            input_seq, input_length)

        predictions = torch.zeros(batch_size, self.max_length + 1)

        for sentence in range(batch_size):
            if encoder_output is not None:
                encoder_output_one = encoder_output[sentence, :, :].unsqueeze(
                    dim=0)
            else:
                encoder_output_one = None
            decoder_hidden_one = decoder_hidden[:, sentence, :].unsqueeze(dim=1)
            decoder_input_one = decoder_input[sentence].unsqueeze(dim=0)
            if context is not None:
                context_one = context[sentence, :, :].unsqueeze(dim=0)
            else:
                context_one = None

            beams_keep = self.beams_init(decoder_input_one,
                                         decoder_hidden_one, encoder_output_one, context_one)

            predict_sen = self.ind_beam(beams_keep, encoder_output_one)
            pred_len = len(predict_sen)
            predict_sen = torch.tensor(predict_sen)

            predictions[sentence, 1:pred_len + 1] = predict_sen

        return predictions


def initialize_model(data, pretrained_embeddings, network_type,
                     embedding_size, hidden_size, num_layers_enc, dropout,
                     bidirectional, kernel_size, num_layers_dec,
                     attention, teacher_forcing_ratio):
    if pretrained_embeddings:
        embeddings_enc = data['train'].input_lang.embeddings
        embeddings_dec = data['train'].target_lang.embeddings
    else:
        embeddings_enc = None
        embeddings_dec = None

    if network_type == 'recurrent':
        encoder = EncoderRNN(input_size=data['train'].input_lang.n_words,
                             embedding_size=embedding_size, hidden_size=hidden_size,
                             num_layers=num_layers_enc, dropout=dropout,
                             bidirectional=bidirectional,
                             pretrained_embeddings=embeddings_enc)
        multiplier = 2 if bidirectional else 1
    elif network_type == 'convolutional':
        encoder = EncoderCNN(
            data['train'].input_lang.n_words, num_layers_enc, embedding_size, hidden_size, kernel_size)
        if attention:
            raise ValueError('Attention is not supported for CNN encoder')
        multiplier = 1

    decoder = DecoderRNN(int(data['train'].target_lang.n_words),
                         embedding_size=embedding_size,
                         hidden_size=(multiplier * hidden_size),
                         num_layers=num_layers_dec,
                         attention=attention,
                         pretrained_embeddings=embeddings_dec)

    return TranslationModel(encoder, decoder,
                            teacher_forcing_ratio=teacher_forcing_ratio).to(model_config.device)
