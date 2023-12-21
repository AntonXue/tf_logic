from typing import Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def default(value, default):
    return default if value is None else value


str_to_activ_module = {
    "relu" : nn.ReLU(),
    "gelu" : nn.GELU()
}

def get_activ(activ_str: str):
    return str_to_activ_module[activ_str]


def e(i, d):
    return torch.eye(d)[:,i].long()


""" Sequence classification models abstract class """

class SeqClsModel(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def model_name(self):
        raise NotImplementedError()

    @property
    def input_dim(self):
        raise NotImplementedError()

    @property
    def embed_dim(self):
        raise NotImplementedError()

    @property
    def num_labels(self):
        raise NotImplementedError()


