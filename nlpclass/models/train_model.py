import argparse
import os.path as osp
from copy import deepcopy

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.utils.data

from nlpclass.models.models import DecoderRNN, EncoderRNN, TranslationModel
from nlpclass.models.training_utils import finalize_run, train_epoch
from tqdm import tqdm

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')
MODEL_DIR = osp.join(CURRENT_PATH, '..', '..', 'models')


def train_model(language, network_type, attention,
                hidden_size, num_layers_enc, num_layers_dec, dropout, bidirectional,
                batch_size, learning_rate, optimizer, n_epochs, early_stopping,
                beam_search, beam_size,
                retrain=False):
    training_parameters = locals()
    data, data_loaders, max_length = load_data(language, batch_size)

    if network_type == 'recurrent':
        encoder = EncoderRNN(data['train']['input_lang'].n_words,
                             hidden_size, num_layers_enc, dropout, bidirectional)
        decoder = DecoderRNN(
            hidden_size, data['train']['output_lang'].n_words, num_layers_dec)
    elif network_type == 'convolutional':
        encoder = None
        decoder = None

    model = TranslationModel(encoder, decoder)

    model_file = f'{network_type}_hidden{hidden_size}_lr{learning_rate}.p'
    if not retrain:
        if osp.exists(osp.join(MODEL_DIR, model_file)):
            model.load_state_dict(torch.load(osp.join(MODEL_DIR, model_file)))
            return model

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

    with mlflow.start_run():
        for par_name, par_value in training_parameters.items():
            mlflow.log_param(par_name, par_value)

        for epoch in tqdm(range(n_epochs)):
            if early_counter >= early_stopping:
                break

            epoch_loss = train_epoch(model, optimizer, data, data_loaders)
            mlflow.log_metric('eval_loss', epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = deepcopy(model)
            else:
                early_counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NLP model')

    parser.add_argument('language', type=str)
    parser.add_argument('--network_type', type=str, default='recurrent')
    parser.add_argument('--attention', action='store_true')

    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers_enc', type=int, default=1)
    parser.add_argument('--num_layers_dec', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bidirectional', action='stor_true')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)

    parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--beam_alpha', type=float, default=0.0)

    parser.add_arguments('--retrain', action='store_true')

    args = parser.parse_args()
    args_dict = vars(args)
