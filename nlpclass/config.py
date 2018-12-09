import torch


class ModelConfig():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_token = 0
    EOS_token = 1
    PAD_token = 2
    grad_norm = 50.0


model_config = ModelConfig()
