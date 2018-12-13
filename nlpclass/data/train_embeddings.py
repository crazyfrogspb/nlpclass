"""
Pretrain word embeddings on the training data
"""


import os.path as osp

from gensim.models import Word2Vec

from nlpclass.data.data_utils import normalizeString
from nlpclass.models.training_utils import load_tokens

CURRENT_PATH = osp.dirname(osp.realpath(__file__))
DATA_DIR = osp.join(CURRENT_PATH, '..', '..', 'data')

lines = {}
lines['vi'], lines['en'] = load_tokens('vi', 'train')
lines['zh'], _ = load_tokens('zh', 'train')

for lang, sentences in lines.items():
    print(f'Training W2V for {lang}')
    sentences = [normalizeString(s).split(' ') for s in sentences]
    model = Word2Vec(sentences, size=256, iter=10,
                     window=5, min_count=1, workers=8)
    model.save(osp.join(DATA_DIR, 'interim', f'model_{lang}.model'))
