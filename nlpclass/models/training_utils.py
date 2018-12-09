import os.path as osp
from copy import deepcopy

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from nlpclass.config import model_config
from nlpclass.data.data_utils import (TranslationDataset, prepareData,
                                      text_collate_func)
from nlpclass.models.models import DecoderRNN, EncoderRNN, TranslationModel

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')
MODEL_DIR = osp.join(CURRENT_PATH, '..', '..', 'models')


def load_tokens(language, dataset_type):
    lang_lines = []
    with open(osp.join(DATA_DIR, 'raw', f'iwslt-{language}-en',
                       f'{dataset_type}.tok.{language}')) as fin:
        for line in fin:
            lang_lines.append(line.strip())

    en_lines = []
    with open(osp.join(DATA_DIR, 'raw', f'iwslt-{language}-en',
                       f'{dataset_type}.tok.en')) as fin:
        for line in fin:
            en_lines.append(line.strip())
    return lang_lines, en_lines


def load_data(language, batch_size):
    data = {}
    data_loaders = {}
    for dataset_type in ['train', 'dev', 'test']:
        lines_en, lines_lang = load_tokens(language, dataset_type)
        data[dataset_type] = TranslationDataset(prepareData(
            language, 'eng', lines_lang, lines_en))
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


def train_epoch(model, optimizer, data, data_loaders):
    model.train()
    for batch in data_loaders['train']:
        total_loss, _, _ = model(batch)
        total_loss.backward()
        clip_grad_norm_(filter(lambda p: p.requires_grad,
                              model.parameters()), model_config.grad_norm)
        optimizer.step()

    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in data_loaders['val']:
            total_loss, _, _ = model(batch)
            epoch_loss += total_loss.item() * \
                batch['input'].size(0) / len(data['val'])

    return epoch_loss


def finalize_run(best_model, best_loss):
    mlflow.log_metric('best_loss', best_loss)
    mlflow.pytorch.log_model(best_model, 'models')


def train_model(language, network_type, attention,
                embedding_size, hidden_size, num_layers_enc, num_layers_dec,
                dropout, bidirectional,
                batch_size, learning_rate, optimizer, n_epochs, early_stopping,
                teacher_forcing_ratio, beam_search, beam_size, beam_alpha,
                retrain=False):
    training_parameters = locals()
    data, data_loaders, max_length = load_data(language, batch_size)

    if network_type == 'recurrent':
        encoder = EncoderRNN(data['train'].input_lang.n_words,
                             embedding_size, hidden_size, num_layers_enc,
                             dropout, bidirectional)
        decoder = DecoderRNN(data['train'].output_lang.n_words,
                             embedding_size, hidden_size, num_layers_dec, attention)
    elif network_type == 'convolutional':
        encoder = None
        decoder = None

    model = TranslationModel(encoder, decoder,
                             max_length=max_length,
                             teacher_forcing_ratio=teacher_forcing_ratio).to(model_config.device)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    else:
        raise ValueError(f'Option {optimizer} is not supported for optimizer')

    best_loss = np.inf
    early_counter = 0
    best_model = deepcopy(model)

    with mlflow.start_run():
        for par_name, par_value in training_parameters.items():
            mlflow.log_param(par_name, par_value)

        for epoch in tqdm(range(n_epochs)):
            if early_counter >= early_stopping:
                finalize_run(best_model, best_loss)
                return best_model

            epoch_loss = train_epoch(model, optimizer, data, data_loaders)
            print(f'Current loss: {epoch_loss}')
            mlflow.log_metric('eval_loss', epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = deepcopy(model)
            else:
                early_counter += 1

        finalize_run(best_model, best_loss)
        return best_model
