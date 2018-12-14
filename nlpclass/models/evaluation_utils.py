"""
Functions to transform predictions to sentences and to calculate BLEU scores
"""

import os
import os.path as osp

import pandas as pd
import sacrebleu
import torch

from nlpclass.config import model_config
from nlpclass.data.load_data import download_model, load_data
from nlpclass.models.models import initialize_model


def output_to_translations(predictions, index2word):
    translations = []
    for row in predictions.cpu().numpy():
        decoded_words = []
        for elem in row:
            if elem not in [model_config.SOS_token, model_config.EOS_token, model_config.PAD_token]:
                decoded_words.append(index2word[elem])
            if elem == model_config.EOS_token:
                break
        translations.append(' '.join(decoded_words))
    return translations


def bleu_eval(ref_trans, new_trans, raw_trans=True):
    # returns a bleu score
    # input lists of strings, must be of the same length!
    if raw_trans:
        return sacrebleu.raw_corpus_bleu(new_trans, [ref_trans]).score
    else:
        return sacrebleu.corpus_bleu(new_trans, [ref_trans]).score


def translate_sentences(row, language_data, cleanup=True):
    download_model(row['Run ID'])

    checkpoint = torch.load(
        osp.join(model_config.model_dir, f"checkpoint_{row['Run ID']}.pth"))

    model = initialize_model(language_data['data'],
                             int(row['pretrained_embeddings']),
                             row['network_type'],
                             int(row['embedding_size']),
                             int(row['hidden_size']),
                             int(row['num_layers_enc']),
                             float(row['dropout']),
                             bool(row['bidirectional']),
                             int(row['kernel_size']),
                             int(row['num_layers_dec']),
                             row['attention'],
                             float(row['teacher_forcing_ratio']))

    model.eval()
    model.load_state_dict(checkpoint['model_state_dict'])

    translations = []

    for sentence in language_data['short_sentences']:
        translation = output_to_translations(
            model.beam_search(
                sentence), checkpoint['target_lang'].index2word)
        translations.append((sentence['target_sentences'], translation))

    for sentence in language_data['long_sentences']:
        translation = output_to_translations(
            model.beam_search(
                sentence), checkpoint['target_lang'].index2word)
        translations.append((sentence['target_sentences'], translation))

    if cleanup:
        os.remove(osp.join(model_config.model_dir,
                           f"checkpoint_{row['Run ID']}.pth"))

    return translations


def print_translations_all(runs_file, sample_size=3, long_threshold=20):
    language_data = {}
    for language in ['vi', 'zh']:
        data, data_loaders = load_data(language, batch_size=1)

        short_sentences = []
        long_sentences = []
        data_iterator = iter(data_loaders['test'])
        while len(short_sentences) < sample_size or len(long_sentences) < sample_size:
            x = next(data_iterator)
            if x['input'].size()[1] < long_threshold:
                if len(short_sentences) < sample_size:
                    short_sentences.append(x)
            else:
                if len(long_sentences) < sample_size:
                    long_sentences.append(x)

        language_data[language] = {
            'data': data, 'data_loaders': data_loaders,
            'short_sentences': short_sentences, 'long_sentences': long_sentences}

    runs_df = pd.read_csv(runs_file)
    for i, row in runs_df.iterrows():
        runs_df.loc[i, 'translations'] = translate_sentences(
            row, language_data[row['language']])
