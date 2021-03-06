"""
Utilities for loading and preprocessing data
"""

import os.path as osp
import re
import unicodedata

import numpy as np
import torch
import torch.utils.data
from gensim.models import KeyedVectors

from nlpclass.config import model_config


class Lang:
    def __init__(self, name):
        """Language class

        Parameters
        ----------
        name : str
            Language name.
        """
        self.name = name
        self.min_count = model_config.min_count
        self.word2index = {"SOS": model_config.SOS_token,
                           "EOS": model_config.EOS_token,
                           "PAD": model_config.PAD_token,
                           "UNK": model_config.UNK_token}
        self.word2count = {}
        self.index2word = {model_config.SOS_token: "SOS",
                           model_config.EOS_token: "EOS",
                           model_config.PAD_token: "PAD",
                           model_config.UNK_token: "UNK"}
        self.n_words = 4
        self.embeddings = None
        self.pretrained_inds = []

    def addSentence(self, sentence):
        # process words from the sentence
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # add word to the language
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self):
        # trim vocabulary
        keep_words = []
        for word, count in self.word2count.items():
            if count >= self.min_count:
                keep_words.append((word, count))

        self.word2index = {"SOS": model_config.SOS_token,
                           "EOS": model_config.EOS_token,
                           "PAD": model_config.PAD_token,
                           "UNK": model_config.UNK_token}
        self.word2count = {}
        self.index2word = {model_config.SOS_token: "SOS",
                           model_config.EOS_token: "EOS",
                           model_config.PAD_token: "PAD",
                           model_config.UNK_token: "UNK"}
        self.n_words = 4  # Count SOS and EOS

        for word, count in keep_words:
            self.addWord(word)
            self.word2count[word] = count

    def load_embeddings(self):
        # load word2vec embeddings
        self.embeddings = np.zeros((self.n_words, model_config.embed_size))
        w2v_embeddings = KeyedVectors.load(
            osp.join(model_config.data_dir, 'interim', f'model_{self.name}.model'))
        for token, id_ in self.word2index.items():
            if token in w2v_embeddings:
                self.embeddings[id_] = w2v_embeddings[token]
                self.pretrained_inds.append(id_)
            elif token == 'PAD':
                self.embeddings[id_] = np.zeros(model_config.embed_size)
                self.pretrained_inds.append(id_)
            else:
                self.embeddings[id_] = np.random.normal(
                    size=(model_config.embed_size, ))


def unicodeToAscii(s):
    # this is not used for now
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # cleaning string
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(" &apos;", r"", s)
    s = re.sub(r"[^\wa-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1_name, lang2_name, lang1_data, lang2_data, reverse=False):
    # Split every line into pairs and normalize
    pairs_pure = zip(lang1_data, lang2_data)
    pairs = [[normalizeString(l[0]), normalizeString(l[1])]
             for l in pairs_pure]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2_name)
        output_lang = Lang(lang1_name)
    else:
        input_lang = Lang(lang1_name)
        output_lang = Lang(lang2_name)

    return input_lang, output_lang, pairs


def prepareData(lang1_name, lang2_name, lang1_data, lang2_data, reverse=False, load_embeddings=False):
    """Prepare data from raw lines.

    Parameters
    ----------
    lang1_name : str
        Name of the input language.
    lang2_name : str
        Name of the target language.
    lang1_data : iter
        Iterable with input sentences.
    lang2_data : iter
        Iterable with target sentences.
    reverse : bool, optional
        Whether to reverse the direction of translation (the default is False).
    load_embeddings : bool, optional
        Whether to load pretrained embeddings (the default is False).

    Returns
    -------
    dict
        Dictionary with input and target language instances and pairs of sentences.

    """
    input_lang, output_lang, pairs = readLangs(
        lang1_name, lang2_name, lang1_data, lang2_data, reverse=reverse)
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    if load_embeddings:
        input_lang.load_embeddings()
        output_lang.load_embeddings()

    input_lang.trim()

    return {'input_lang': input_lang, 'output_lang': output_lang, 'pairs': pairs}


def indexesFromSentence(lang, sentence):
    # convert tokens to indices
    tokens = sentence.split(' ')[:model_config.max_length]
    indices = [lang.word2index.get(word, model_config.UNK_token)
               for word in tokens]
    indices.append(model_config.EOS_token)
    return indices


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, input_lang, target_lang, pairs):
        """Translation Dataset class.

        Parameters
        ----------
        input_lang : Lang
            Lang instance of the input language.
        target_lang : Lang
            Lang instance of the target language.
        pairs : iter
            Iterable with pairs of the sentences.
        """
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_indices = indexesFromSentence(self.input_lang, self.pairs[idx][0])
        target_indices = indexesFromSentence(
            self.target_lang, self.pairs[idx][1])
        target_sentence = self.pairs[idx][1]

        return input_indices, target_indices, target_sentence


def pad_seq(seq, max_length):
    # pad sequences
    seq += [model_config.PAD_token for i in range(max_length - len(seq))]
    return seq


def text_collate_func(batch):
    # adapted from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
    # dynamically sorts a batch in descending order of the length of the input sequences
    seq_pairs = sorted(batch,
                       key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs, target_sentences = zip(*seq_pairs)

    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(seq, max(input_lengths)) for seq in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(seq, max(target_lengths)) for seq in target_seqs]

    return {'input': torch.LongTensor(input_padded).to(model_config.device),
            'target': torch.LongTensor(target_padded).to(model_config.device),
            'input_length': torch.LongTensor(input_lengths).to(model_config.device),
            'target_length': torch.LongTensor(target_lengths).to(model_config.device),
            'target_sentences': target_sentences}
