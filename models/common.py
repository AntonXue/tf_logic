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
    return torch.eye(d)[:,i]


class MySeq2SeqModel(nn.Module):
    def __init__(self, embed_dim: int, max_seq_len: Optional[int] = None):
        """ embed_dim: the embedding dimension
            max_seq_len: the forward function usually assumes a tensor of shape
                            x: (batch_size, seq_len, embed_dim)
                        if max_seq_len is None, then there is no bound on seq_len.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len


class MySeqClsModel(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_labels: int,
            max_seq_len: Optional[int] = None,
            problem_type: Optional[str] = None):
        """ embed_dim: the embedding dimension
            num_labels: how many output labels
            max_seq_len: how many
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.max_seq_len = max_seq_len
        self.problem_type = problem_type


