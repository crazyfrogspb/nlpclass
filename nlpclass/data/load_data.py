import os.path as osp
import random

import torch.utils.data

from nlpclass.data.data_utils import (TranslationDataset, prepareData,
                                      text_collate_func)

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')


def load_tokens(language, dataset_type):
    # load tokenized sentences for the language
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
    # create dataset and data loader instances
    data = {}
    data_loaders = {}
    for dataset_type in ['train', 'dev', 'test']:
        lines_lang, lines_en = load_tokens(language, dataset_type)

        if subsample < 1.0 and dataset_type == 'train':
            # for testing
            sample_size = int(subsample * len(lines_en))
            lines_lang, lines_en = zip(
                *random.sample(list(zip(lines_lang, lines_en)), sample_size))

        load_embeddings = True if dataset_type == 'train' else False
        data_dict = prepareData(language, 'en', lines_lang,
                                lines_en, load_embeddings=load_embeddings)

        if dataset_type == 'train':
            data[dataset_type] = TranslationDataset(
                data_dict['input_lang'], data_dict['output_lang'], data_dict['pairs'])
        else:
            # use train Lang instance
            data[dataset_type] = TranslationDataset(
                data['train'].input_lang, data['train'].target_lang, data_dict['pairs'])

        data_loaders[dataset_type] = torch.utils.data.DataLoader(dataset=data[dataset_type],
                                                                 batch_size=batch_size,
                                                                 collate_fn=text_collate_func,
                                                                 shuffle=True)

    return data, data_loaders
