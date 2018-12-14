"""
All utilities for training the model and tracking its quality
"""

import os.path as osp
from copy import deepcopy

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.utils.data
from torch.nn.utils import clip_grad_norm_

from nlpclass.config import model_config
from nlpclass.data.load_data import load_data
from nlpclass.models.evaluation_utils import bleu_eval, output_to_translations
from nlpclass.models.models import initialize_model

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')
MODEL_DIR = osp.join(CURRENT_PATH, '..', '..', 'models')


def train_epoch(model, optimizer, scheduler, clipping_value, data, data_loaders):
    # train model for one epoch
    epoch_loss = 0
    for i, batch in enumerate(data_loaders['train']):
        model.train()
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        clip_grad_norm_(filter(lambda p: p.requires_grad,
                               model.parameters()), clipping_value)
        optimizer.step()
        epoch_loss += loss.item()

        if i % model_config.logging_freq == 0:
            val_loss, val_bleu = evaluate(model, data, data_loaders)
            if scheduler is not None:
                scheduler.step(val_loss)
            mlflow.log_metric('val_loss', val_loss)
            mlflow.log_metric('val_bleu', val_bleu)

    return epoch_loss / (i + 1)


def finalize_run(best_model, best_bleu, best_loss):
    # finalize run
    mlflow.log_metric('best_loss', best_loss)
    mlflow.log_metric('best_bleu', best_bleu)


def evaluate(model, data, data_loaders, dataset_type='dev', max_batch=100, greedy=False):
    model.eval()
    epoch_loss = 0
    target_index2word = data['train'].target_lang.index2word
    with torch.no_grad():
        original_strings = []
        translated_strings = []
        for i, batch in enumerate(data_loaders[dataset_type]):
            if i > max_batch:
                break
            loss = model(batch)
            epoch_loss += loss.item()
            original = batch['target_sentences']
            if greedy:
                translations = output_to_translations(
                    model.greedy(batch), target_index2word)
            else:
                translations = output_to_translations(
                    model.beam_search(batch), target_index2word)
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
    # main function for training the model
    training_parameters = locals()

    data, data_loaders = load_data(language, subsample, batch_size)

    model = initialize_model(data, pretrained_embeddings, network_type,
                             embedding_size, hidden_size, num_layers_enc, dropout,
                             bidirectional, kernel_size, num_layers_dec,
                             attention, teacher_forcing_ratio)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), learning_rate, momentum=0.9)
    else:
        raise ValueError(f'Option {optimizer} is not supported for optimizer')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=model_config.decay_patience, factor=model_config.decay_factor)

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
                return 'success'

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
                early_counter = 0
                best_loss = val_loss
                best_bleu = val_bleu
                best_model = deepcopy(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'input_lang': data['train'].input_lang,
                    'target_lang': data['train'].target_lang
                }, osp.join(MODEL_DIR, f'checkpoint_{mlflow.active_run()._info.run_uuid}.pth'))
            else:
                early_counter += 1

        return 'success'
