import os.path as osp
import random
import warnings
from copy import deepcopy

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.utils.data
from torch.nn.utils import clip_grad_norm_

from nlpclass.config import model_config
from nlpclass.data.data_utils import (TranslationDataset, prepareData,
                                      text_collate_func)
from nlpclass.models.evaluation_utils import bleu_eval, output_to_translations
from nlpclass.models.models import (DecoderRNN, EncoderCNN, EncoderRNN,
                                    TranslationModel)

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')
MODEL_DIR = osp.join(CURRENT_PATH, '..', '..', 'models')


def load_tokens(language, dataset_type):
    lines_lang = []
    with open(osp.join(DATA_DIR, 'raw', f'iwslt-{language}-en',
                       f'{dataset_type}.tok.{language}')) as fin:
        for line in fin:
            lines_lang.append(line.strip())

    lines_en = []
    with open(osp.join(DATA_DIR, 'raw', f'iwslt-{language}-en',
                       f'{dataset_type}.tok.en')) as fin:
        for line in fin:
            lines_en.append(line.strip())
    return lines_lang, lines_en


def load_data(language, subsample=1.0, batch_size=16):
    data = {}
    data_loaders = {}
    for dataset_type in ['train', 'dev', 'test']:
        lines_lang, lines_en = load_tokens(language, dataset_type)
        if subsample < 1.0 and dataset_type == 'train':
            sample_size = int(subsample * len(lines_en))
            lines_lang, lines_en = zip(
                *random.sample(list(zip(lines_lang, lines_en)), sample_size))
        if dataset_type == 'train':
            load_embeddings = True
        else:
            load_embeddings = False
        data_dict = prepareData(language, 'en', lines_lang,
                                lines_en, load_embeddings=load_embeddings)
        if dataset_type == 'train':
            data[dataset_type] = TranslationDataset(
                data_dict['input_lang'], data_dict['output_lang'], data_dict['pairs'])
        else:
            data[dataset_type] = TranslationDataset(
                data['train'].input_lang, data['train'].target_lang, data_dict['pairs'])
        data_loaders[dataset_type] = torch.utils.data.DataLoader(dataset=data[dataset_type],
                                                                 batch_size=batch_size,
                                                                 collate_fn=text_collate_func,
                                                                 shuffle=True)

    max_length = 0
    for x in data['train'].pairs:
        len1 = len(x[0].split(" "))
        len2 = len(x[1].split(" "))
        max_len = max(len1, len2)
        if max_len > max_length:
            max_length = max_len

    return data, data_loaders, max_length

def train_epoch(model, optimizer, scheduler, clipping_value, data, data_loaders, logging_freq=500):
    epoch_loss = 0
    for i, batch in enumerate(data_loaders['train']):
        model.train()
        optimizer.zero_grad()
        loss = model(batch)
        #loss = calc_loss(logits, batch['target'], criterion)
        #print(total_loss, loss)
        loss.backward()
        clip_grad_norm_(filter(lambda p: p.requires_grad,
                                model.parameters()), clipping_value)
        optimizer.step()
        epoch_loss += loss.item()
        if i % logging_freq == 0:
            val_loss, val_bleu = evaluate(model, data, data_loaders)
            if scheduler is not None:
                scheduler.step(val_loss)
            mlflow.log_metric('val_loss', val_loss)
            mlflow.log_metric('val_bleu', val_bleu)

            # train_loss, train_bleu = evaluate(
            #    model, data, data_loaders, criterion, dataset_type='train')
            #mlflow.log_metric('train_loss', train_loss)
            #mlflow.log_metric('train_bleu', train_bleu)
    return epoch_loss / (i + 1)


def finalize_run(best_model, best_bleu, best_loss):
    mlflow.log_metric('best_loss', best_loss)
    mlflow.log_metric('best_bleu', best_bleu)
    mlflow.pytorch.log_model(best_model, 'models')


def evaluate(model, data, data_loaders, dataset_type='dev', max_batch=100, greedy=True):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        original_strings = []
        translated_strings = []
        for i, batch in enumerate(data_loaders[dataset_type]):
            if i > max_batch:
                break
            loss = model(batch)
            epoch_loss += loss.item()
            original = output_to_translations(batch['target'], data['train'])
            if greedy:
                translations = output_to_translations(
                    model.greedy(batch), data['train'])
            else:
                translations = output_to_translations(
                    model.beam_search(batch), data['train'])
            original_strings.extend(original)
            translated_strings.extend(translations)
        bleu = bleu_eval(original_strings, translated_strings)
        model.train()

    return epoch_loss / (i + 1), bleu


def train_model(language, network_type, attention,
                embedding_size, hidden_size, num_layers_enc, num_layers_dec,
                dropout, bidirectional,
                batch_size, learning_rate, optimizer, clipping_value,
                n_epochs, early_stopping, pretrained_embeddings,
                teacher_forcing_ratio, beam_search, beam_size, beam_alpha,
                subsample, kernel_size):
    training_parameters = locals()
    data, data_loaders, max_length = load_data(language, subsample, batch_size)

    if network_type == 'recurrent':
        if pretrained_embeddings:
            embeddings_enc = data['train'].input_lang.embeddings
            embeddings_dec = data['train'].target_lang.embeddings
        else:
            embeddings_enc = None
            embedding_dec
        encoder = EncoderRNN(input_size=data['train'].input_lang.n_words,
                             embedding_size=embedding_size, hidden_size=hidden_size,
                             num_layers=num_layers_enc, dropout=dropout,
                             bidirectional=bidirectional,
                             pretrained_embeddings=embeddings_enc)
        if bidirectional:
            multiplier = 2
        else:
            multiplier = 1
        decoder = DecoderRNN(data['train'].target_lang.n_words,
                             embedding_size=embedding_size,
                             hidden_size=(multiplier * hidden_size),
                             num_layers=num_layers_dec,
                             attention=attention,
                             pretrained_embeddings=embeddings_dec)
    elif network_type == 'convolutional':
        encoder = EncoderCNN(
            data['train'].input_lang.n_words, num_layers_enc, embedding_size, hidden_size, kernel_size)
        if attention:
            warnings.warn('Attention is not supported for CNN encoder')
        decoder = DecoderRNN(data['train'].target_lang.n_words,
                             embedding_size, hidden_size, num_layers_dec, attention=False)

    model = TranslationModel(encoder, decoder,
                             teacher_forcing_ratio=teacher_forcing_ratio).to(model_config.device)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
        scheduler = None
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=2)
    else:
        raise ValueError(f'Option {optimizer} is not supported for optimizer')

    weight = torch.ones(model.decoder.output_size).to(model_config.device)
    weight[model_config.PAD_token] = 0

    best_bleu = 0.0
    best_loss = np.inf
    early_counter = 0
    best_model = deepcopy(model)

    with mlflow.start_run():
        for par_name, par_value in training_parameters.items():
            mlflow.log_param(par_name, par_value)

        for epoch in range(n_epochs):
            print(f'Fitting epoch {epoch}')
            if early_counter >= early_stopping:
                finalize_run(best_model, best_bleu, best_loss)
                return best_model

            train_loss = train_epoch(
                model, optimizer, scheduler, clipping_value, data, data_loaders)
            mlflow.log_metric('train_loss_epoch', train_loss)

            val_loss, val_bleu_greedy = evaluate(
                model, data, data_loaders)
            mlflow.log_metric('val_loss_epoch', val_loss)
            mlflow.log_metric('val_bleu_greedy', val_bleu_greedy)
            val_loss, val_bleu_beam = evaluate(
                model, data, data_loaders, greedy=False)
            mlflow.log_metric('val_bleu_beam', val_bleu_beam)

            val_bleu = max(val_bleu_greedy, val_bleu_beam)

            if val_bleu >= best_bleu:
                best_loss = val_loss
                best_bleu = val_bleu
                best_model = deepcopy(model)
            else:
                early_counter += 1

        finalize_run(best_model, best_bleu, best_loss)
        return best_model
