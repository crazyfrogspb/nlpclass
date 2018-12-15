import os.path as osp

from nlpclass.config import model_config
from nlpclass.data.load_data import load_tokens

LONG_THRESHOLD = 20

for language in ['zh', 'vi']:
    lines_lang, lines_en = load_tokens('zh', 'dev')
    lines_lang = [line.split(' ') for line in lines_lang]
    lines_en = [line.split(' ') for line in lines_en]

    lines_zh_short = [line for line in lines_lang if len(line) < LONG_THRESHOLD]
    lines_en_short = [line for line in lines_en if len(line) < LONG_THRESHOLD]
    lines_zh_long = [line for line in lines_lang if len(line) >= LONG_THRESHOLD]
    lines_en_long = [line for line in lines_en if len(line) >= LONG_THRESHOLD]

    with open(osp.join(model_config.data_dir, f'iwslt-{language}-en', 'dev_short.tok.en', 'w')) as fout:
        for line in lines_en_short:
            fout.write("%s\n" % ' '.join(line))
    with open(osp.join(model_config.data_dir, f'iwslt-{language}-en', f'dev_short.tok.{language}', 'w'), 'w') as fout:
        for line in lines_zh_short:
            fout.write("%s\n" % ' '.join(line))
    with open(osp.join(model_config.data_dir, f'iwslt-{language}-en', 'dev_long.tok.en', 'w'), 'w') as fout:
        for line in lines_en_long:
            fout.write("%s\n" % ' '.join(line))
    with open(osp.join(model_config.data_dir, f'iwslt-{language}-en', 'dev_long.tok.{language}', 'w'), 'w') as fout:
        for line in lines_zh_long:
            fout.write("%s\n" % ' '.join(line))
