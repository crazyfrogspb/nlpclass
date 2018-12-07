import argparse
import os.path as osp

import mlflow

from nlpclass.data.data_utils import prepareData

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NLP model')

    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--beam', action='store_true')
    parser.add_argument('--beamsize', type=int, default=5)

    args = parser.parse_args()
    args_dict = vars(args)

    with mlflow.start_run():
        mlflow.log_param('attention', args_dict['attention'])
        mlflow.log_param('beam_search', args_dict['beam'])
        mlflow.log_param('beam_size', args_dict['beamsize'])
