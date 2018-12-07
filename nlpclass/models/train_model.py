import argparse
import os.path as osp

import mlflow
import torch

from nlpclass.data.data_utils import prepareData
from nlpclass.models.models import DecoderRNN, EncoderRNN

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')
MODEL_DIR = osp.join(CURRENT_PATH, '..', '..', 'models')


def load_tokens(filename):
    lines = []
    with open(osp.join(DATA_DIR, 'raw', filename)) as fin:
        for line in fin:
            lines.append(line.strip())
    return lines


def load_data():
    lines_en_train = load_tokens('train.tok.en')
    lines_zh_train = load_tokens('train.tok.zh')
    lines_en_dev = load_tokens('dev.tok.en')
    lines_zh_dev = load_tokens('dev.tok.zh')

    train_data = {}
    train_data['input_lang'], train_data['output_lang'], train_data['pairs'] = \
        prepareData('chin', 'eng', lines_zh_train, lines_en_train)
    dev_data = {}
    dev_data['input_lang'], train_data['output_lang'], train_data['pairs'] = \
        prepareData('chin', 'eng', lines_zh_dev, lines_en_dev)

    max_length = 0
    for x in train_data['pairs']:
        len1 = len(x[0].split(" "))
        len2 = len(x[1].split(" "))
        max_len = max(len1, len2)
        if max_len > max_length:
            max_length = max_len

    return train_data, dev_data, max_length


def train_model(network_type, hidden_size, learning_rate, retrain=False):
    train_data, dev_data, max_length = load_data()

    if network_type == 'basic':
        pass
    elif network_type == 'attention':
        pass
    elif network_type == 'convolutional':
        pass

    model_file = f'{network_type}_hidden{hidden_size}_lr{learning_rate}.p'
    if not retrain:
        if osp.exists(osp.join(MODEL_DIR, model_file)):
            model.load_state_dict(torch.load(osp.join(MODEL_DIR, model_file)))
            return model

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NLP model')

    parser.add_argument('--network_type', type=str, default='basic')
    parser.add_arguments('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_arguments('--retrain', action='store_true')

    args = parser.parse_args()
    args_dict = vars(args)

    with mlflow.start_run():
        mlflow.log_param('attention', args_dict['attention'])
        mlflow.log_param('beam_search', args_dict['beam_search'])
        mlflow.log_param('beam_size', args_dict['beams_ize'])
        mlflow.log_param('hidden_size', args_dict['hidden_size'])
