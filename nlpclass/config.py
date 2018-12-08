import torch


class ModelConfig():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_token = 0
    EOS_token = 1
    PAD_token = 2
    teacher_forcing_ratio = 0.5


model_config = ModelConfig()
