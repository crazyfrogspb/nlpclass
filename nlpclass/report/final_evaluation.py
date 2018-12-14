"""
FInal evaluations
"""
import argparse
import os
import os.path as osp

import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid

from nlpclass.config import model_config
from nlpclass.data.load_data import download_model, load_data
from nlpclass.models.evaluation_utils import evaluate, output_to_translations
from nlpclass.models.models import initialize_model


def tune_beam(model, data, data_loaders, tuning_dict=None):
    if tuning_dict is None:
        tuning_dict = {'beam_size': list(range(1, 9)),
                       'beam_alpha': [0.0, 0.5, 1.0]}

    tuning_grid = ParameterGrid(tuning_dict)

    best_bleu = 0.0
    best_parameters = None
    for parameters in tuning_grid:
        model.beam_size = parameters['beam_size']
        model.beam_alpha = parameters['beam_alpha']
        _, bleu = evaluate(model, data, data_loaders, greedy=False)
        print(f'BLEU score for parameters {parameters}: {bleu}')
        if bleu > best_bleu:
            best_bleu = bleu
            best_parameters = parameters

    return best_parameters, best_bleu


def generate_translations(model, language_data, checkpoint):
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

    return translations


def evaluate_model(row, language_data, tune_beam=False, evaluate_test=False, cleanup=True):
    download_model(row['Run ID'])
    results = {}

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

    if tune_beam:
        results['best_parameters'] = tune_beam(
            model, language_data['data'], language_data['data_loaders'])
        model.beam_size = results['best_parameters']['beam_size']
        model.beam_alpha = results['best_parameters']['beam_alpha']

    if evaluate_test:
        results['test_loss'], results['test_bleu'] = evaluate(
            model, language_data['data'], language_data['data_loaders'], dataset_type='test')

    results['translations'] = generate_translations(
        model, language_data, checkpoint)

    if cleanup:
        os.remove(osp.join(model_config.model_dir,
                           f"checkpoint_{row['Run ID']}.pth"))

    return results


def evaluate_all_models(language, runs_file, output_file, sample_size=3, long_threshold=20):
    runs_df = pd.read_csv(runs_file)

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

    best_model_id = runs_df.loc[runs_df['best_bleu'].idxmax(), 'Run ID']

    language_data = {
        'data': data, 'data_loaders': data_loaders,
        'short_sentences': short_sentences, 'long_sentences': long_sentences}

    for i, row in runs_df.iterrows():
        if row['Run ID'] == best_model_id:
            results = evaluate_model(
                row, language_data, tune_beam=True, evaluate_test=True)
        else:
            results = evaluate_model(row, language_data)
        for key, value in results.items():
            runs_df.loc[i, key] = value

    runs_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate NLP models')

    parser.add_argument('language', type=str)
    parser.add_argument('runs_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--sample_size', type=int, default=3)
    parser.add_argument('--long_threshold', type=int, default=20)

    args = parser.parse_args()
    args_dict = vars(args)

    evaluate_all_models(**args_dict)
