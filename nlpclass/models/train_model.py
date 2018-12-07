import argparse
import os.path as osp

import mlflow
import torch

from nlpclass.data.data_utils import prepareData
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


def load_data(language):
    lines_en_train, lines_lang_train = load_tokens(language, 'train')
    lines_en_dev, lines_lang_dev = load_tokens(language, 'dev')
    lines_en_test, lines_lang_test = load_tokens(language, 'test')

    data = {}
    for dataset_type in ['train', 'dev', 'test']:
        lines_en, lines_lang = load_tokens(language, dataset_type)
        data[dataset_type] = prepareData(language, 'eng', lines_lang, lines_en)

    max_length = 0
    for x in data['train']['pairs']:
        len1 = len(x[0].split(" "))
        len2 = len(x[1].split(" "))
        max_len = max(len1, len2)
        if max_len > max_length:
            max_length = max_len

    return data, max_length


def train_model(language, network_type,
                hidden_size, num_layers_enc, num_layers_dec, dropout, bidirectional,
                learning_rate, optimizer, n_epochs, early_stopping,
                beam_search, beam_size,
                retrain=False):
    data, max_length = load_data(language)

    if network_type == 'basic':
        encoder = EncoderRNN(data['train']['input_lang'].n_words, hidden_size)
        decoder = DecoderRNN(hidden_size, data['train']['output_lang'].n_words)
    elif network_type == 'attention':
        encoder = None
        decoder = None
    elif network_type == 'convolutional':
        encoder = None
        decoder = None

    model = TranslationModel(encoder, decoder)

    model_file = f'{network_type}_hidden{hidden_size}_lr{learning_rate}.p'
    if not retrain:
        if osp.exists(osp.join(MODEL_DIR, model_file)):
            model.load_state_dict(torch.load(osp.join(MODEL_DIR, model_file)))
            return model

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NLP model')

    parser.add_argument('language', type=str)
    parser.add_argument('--network_type', type=str, default='basic')

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers_enc', type=int, default=1)
    parser.add_argument('--num_layers_dec', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bidirectional', action='stor_true')

    parser.add_arguments('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)

    parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--beam_size', type=int, default=5)

    parser.add_arguments('--retrain', action='store_true')

    args = parser.parse_args()
    args_dict = vars(args)

    with mlflow.start_run():
        for arg, arg_value in args_dict.items():
            mlflow.log_param(arg, arg_value)
