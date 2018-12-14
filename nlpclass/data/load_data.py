"""
Utilities for loading data and downloading models from S3 bucket
"""

import os.path as osp
import random

import boto3
import botocore
import torch.utils.data

from nlpclass.config import model_config
from nlpclass.data.data_utils import (TranslationDataset, prepareData,
                                      text_collate_func)


def load_tokens(language, dataset_type):
    # load tokenized sentences for the language
    lines_lang = []
    with open(osp.join(model_config.data_dir, 'raw', f'iwslt-{language}-en',
                       f'{dataset_type}.tok.{language}')) as fin:
        for line in fin:
            lines_lang.append(line.strip())

    lines_en = []
    with open(osp.join(model_config.data_dir, 'raw', f'iwslt-{language}-en',
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
            shuffle = True
        else:
            # use train Lang instance
            data[dataset_type] = TranslationDataset(
                data['train'].input_lang, data['train'].target_lang, data_dict['pairs'])
            shuffle = False

        data_loaders[dataset_type] = torch.utils.data.DataLoader(dataset=data[dataset_type],
                                                                 batch_size=batch_size,
                                                                 collate_fn=text_collate_func,
                                                                 shuffle=shuffle)

    return data, data_loaders


def download_model(run_uuid):
    if not osp.exists(osp.join(model_config.model_dir, f"checkpoint_{run_uuid}.pth")):
        s3 = boto3.resource('s3')
        try:
            s3.Bucket('nikitinphd').download_file(
                f"nlp/models/checkpoint_{run_uuid}.pth",
                osp.join(model_config.model_dir, f"checkpoint_{run_uuid}.pth"))
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise ValueError('The model with this id does not exist')
            else:
                raise
