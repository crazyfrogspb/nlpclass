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
        translations.append(' '.join(decoded_words))
    return translations


def bleu_eval(ref_trans, new_trans, raw_trans=True):
    # returns a bleu score
    # input lists of strings, must be of the same length!
    if raw_trans:
        return sacrebleu.raw_corpus_bleu(new_trans, [ref_trans]).score
    else:
        return sacrebleu.corpus_bleu(new_trans, [ref_trans]).score


def print_translations(runs_file, experiment_id, sample_size=3, long_threshold=20):
    language_data = {}
    for language in ['vi', 'zh']:
        data, data_loaders = load_data(language, batch_size=1)

        short_sentences = []
        long_sentences = []
        data_iterator = iter(data_loaders['test'])
        while len(short_sentences) < sample_size and len(long_sentences) < sample_size:
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
        checkpoint = torch.load(
            osp.join(MODEL_DIR, f"checkpoint_{row['Run ID']}.pth"))

        model = initialize_model(language_data[row['language'].item()]['data'],
                                 row['pretrained_embeddings'].item(),
                                 row['network_type'].item(),
                                 row['embedding_size'].item(),
                                 row['hidden_size'].item(),
                                 row['num_layers_enc'].item(),
                                 row['dropout'].item(),
                                 row['bidirectional'].item(),
                                 row['kernel_size'].item(),
                                 row['num_layers_dec'].item(),
                                 row['attention'].item(),
                                 row['teacher_forcing_ratio'].item())

        model.eval()
        model.load_state_dict(checkpoint['model_state_dict'])

        for sentence in short_sentences:
            original = output_to_translations(
                sentence['input'], checkpoint['input_lang_w2i'], checkpoint['target_lang_w2i'])
            translation = output_to_translations(
                model.beam_search(
                    sentence), checkpoint['input_lang_w2i'], checkpoint['target_lang_w2i'])
            print(original, translation)
