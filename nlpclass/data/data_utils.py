import re
import unicodedata

from nlpclass.config import model_config


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {model_config.SOS_token: "SOS",
                           model_config.EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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
    #this is not used for now
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    #s = unicodeToAscii(s.lower().strip())
    s = s.lower().strip()
    s = re.sub(r"([!?])", r" \1", s)
    s = re.sub(" &apos;",r"", s)
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
