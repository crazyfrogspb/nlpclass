import os.path as osp

import torch

CURRENT_PATH = osp.dirname(osp.realpath(__file__))


class ModelConfig():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_token = 0
    EOS_token = 1
    PAD_token = 2
    UNK_token = 3
    min_count = 2
    max_length = 100
    embed_size = 256
    embedding_init = 0.05
    logging_freq = 500
    decay_patience = 5
    decay_factor = 0.1
    data_dir = osp.join(CURRENT_PATH, '..', 'data')
    model_dir = osp.join(CURRENT_PATH, '..', 'models')
    bucket_name = 'nikitinphd'


model_config = ModelConfig()
