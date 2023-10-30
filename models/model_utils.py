from typing import Optional

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
    """ embed_dim: the embedding dimensional
        max_seq_len: the forward function usually assumes a tensor of shape
                        x: (seq_len, batch_size, embed_dim)
                    if max_seq_len is None, then there is no bound on seq_len.
    """
    def __init__(
            self,
            embed_dim: int,
            max_seq_len: Optional[int] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

