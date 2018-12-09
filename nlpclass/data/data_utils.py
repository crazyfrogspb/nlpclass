import re
import unicodedata

import torch
import torch.utils.data

from nlpclass.config import model_config


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {model_config.SOS_token: "SOS",
                           model_config.EOS_token: "EOS",
                           model_config.PAD_token: "PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    # this is not used for now
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    #s = unicodeToAscii(s.lower().strip())
    s = s.lower().strip()
    s = re.sub(r"([!?])", r" \1", s)
    s = re.sub(" &apos;", r"", s)
    s = re.sub(r"[^\wa-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1_name, lang2_name, lang1_data, lang2_data, reverse=False):
    # Split every line into pairs and normalize
    pairs_pure = zip(lang1_data, lang2_data)
    pairs = [[normalizeString(l[0]), normalizeString(l[1])] for l in pairs_pure]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2_name)
        output_lang = Lang(lang1_name)
    else:
        input_lang = Lang(lang1_name)
        output_lang = Lang(lang2_name)

    return input_lang, output_lang, pairs


def prepareData(lang1_name, lang2_name, lang1_data, lang2_data, reverse=False):
    input_lang, output_lang, pairs = readLangs(
        lang1_name, lang2_name, lang1_data, lang2_data)
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return {'input_lang': input_lang, 'output_lang': output_lang, 'pairs': pairs}


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(model_config.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=model_config.device).view(-1, 1)


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.input_lang = data['input_lang']
        self.output_lang = data['output_lang']
        self.pairs = data['pairs']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_indices = indexesFromSentence(self.input_lang, self.pairs[idx][0])
        output_indices = indexesFromSentence(
            self.output_lang, self.pairs[idx][1])

        return input_indices, output_indices


def pad_seq(seq, max_length):
    seq += [model_config.PAD_token for i in range(max_length - len(seq))]
    return seq


def text_collate_func(batch):
    # thanks to https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
    unzipped = list(zip(*batch))
    input_seqs = unzipped[0]
    target_seqs = unzipped[1]

    seq_pairs = sorted(zip(input_seqs, target_seqs),
                       key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    return {'input': torch.LongTensor(input_padded).to(model_config.device),
            'target': torch.LongTensor(target_padded).to(model_config.device),
            'input_length': torch.LongTensor(input_lengths).to(model_config.device),
            'output_length': torch.LongTensor(target_lengths).to(model_config.device)}
