import torch
import torch.nn as nn
import transformers


def default(value, default):
    return default if value is None else value


str_to_activ_module = {
    "relu" : nn.ReLU(),
    "gelu" : nn.GELU()
}

def get_activ(activ_str):
    return str_to_activ_module[activ_str]


class MySeq2SeqModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

