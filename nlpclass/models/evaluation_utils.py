"""
Functions to transform predictions to sentences and to calculate BLEU scores
"""

import sacrebleu

from nlpclass.config import model_config


def output_to_translations(predictions, data, output=True):
    translations = []
    for row in predictions.cpu().numpy():
        decoded_words = []
        for elem in row:
            if elem not in [model_config.SOS_token, model_config.EOS_token, model_config.PAD_token]:
                if output:
                    decoded_words.append(data.output_lang.index2word[elem])
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
