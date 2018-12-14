"""
Functions to transform predictions to sentences and to calculate BLEU scores
"""
import numpy as np
import sacrebleu
import torch

from nlpclass.config import model_config


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


def evaluate(model, data, data_loaders, dataset_type='dev', max_batch=None, greedy=True):
    if max_batch is None:
        max_batch = np.inf
    model.eval()
    epoch_loss = 0
    target_index2word = data['train'].target_lang.index2word
    with torch.no_grad():
        original_strings = []
        translated_strings = []
        for i, batch in enumerate(data_loaders[dataset_type]):
            if i > max_batch:
                break
            loss = model(batch)
            epoch_loss += loss.item()
            original = batch['target_sentences']
            if greedy:
                translations = output_to_translations(
                    model.greedy(batch), target_index2word)
            else:
                translations = output_to_translations(
                    model.beam_search(batch), target_index2word)
            original_strings.extend(original)
            translated_strings.extend(translations)
        bleu = bleu_eval(original_strings, translated_strings)
        model.train()

    return epoch_loss / (i + 1), bleu
