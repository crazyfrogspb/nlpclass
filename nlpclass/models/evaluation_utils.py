"""
Functions to transform predictions to sentences and to calculate BLEU scores
"""

import os.path as osp

import pandas as pd
import sacrebleu
import torch

from nlpclass.config import model_config
from nlpclass.data.load_data import load_data
from nlpclass.models.models import initialize_model

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
MODEL_DIR = osp.join(CURRENT_PATH, '..', '..', 'models')


def output_to_translations(predictions, data, output=True):
    translations = []
    for row in predictions.cpu().numpy():
        decoded_words = []
        for elem in row:
            if elem not in [model_config.SOS_token, model_config.EOS_token, model_config.PAD_token]:
                if output:
                    decoded_words.append(data.target_lang.index2word[elem])
                else:
                    decoded_words.append(data.input_lang.index2word[elem])
            if elem == model_config.EOS_token:
                break
                translations.append(' '.join(decoded_words))
        translations.append(' '.join(decoded_words))
    return translations


def bleu_eval(ref_trans, new_trans, raw_trans=True):
    # returns a bleu score
    # input lists of strings, must be of the same length!
    if raw_trans:
        return sacrebleu.raw_corpus_bleu(new_trans, [ref_trans]).score
    else:
        return sacrebleu.corpus_bleu(new_trans, [ref_trans]).score


def print_translations(runs_file, experiment_id, batch_size=32):
    language_data = {}
    for language in ['vi', 'zh']:
        data, data_loaders = load_data(language, batch_size=32)

    runs_df = pd.read_csv(runs_file)
    run_ids = runs_df['Run ID'].unique()
    for run_id in run_ids:
        checkpoint = torch.load(osp.join(MODEL_DIR, f'checkpoint_{run_id}.pth'))

        model = initialize_model(data, pretrained_embeddings, network_type,
                                 embedding_size, hidden_size, num_layers_enc,
                                 dropout, bidirectional, kernel_size,
                                 num_layers_dec, attention, teacher_forcing_ratio)
