from typing import Optional

import copy
import math
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from .model_utils import *


""" Our custom transformer model
"""

class MyTfConfig:
    """ The configs
    """
    def __init__(self, embed_dim, ffwd_width, ffwd_depth, num_heads, num_layers,
                 ffwd_activ = "relu",
                 do_norm = True,
                 layer_norm_epsilon = 1e-5):
        self.embed_dim = embed_dim
        self.ffwd_width = ffwd_width
        self.ffwd_depth = ffwd_depth
        self.ffwd_activ = ffwd_activ
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.do_norm = do_norm
        self.layer_norm_epsilon = layer_norm_epsilon


class AFBlock(nn.Module):
    """ A single attention-feedforward block
    """
    def __init__(self, config: MyTfConfig):
        super().__init__()
        # Norm layers
        if config.do_norm:
            self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_epsilon)
            self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_epsilon)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Attention block
        self.attn = nn.MultiheadAttention(config.embed_dim, config.num_heads)

        # Feedforward block construction
        ffwd_parts = [nn.Linear(config.embed_dim, config.ffwd_width), get_activ(config.ffwd_activ)]
        for i in range(config.ffwd_depth-1):
            ffwd_parts.append(nn.Linear(config.ffwd_width, config.ffwd_width))
            ffwd_parts.append(get_activ(config.ffwd_activ))
        ffwd_parts.append(nn.Linear(config.ffwd_width, config.embed_dim))
        self.ffwd = nn.Sequential(*ffwd_parts)

    def forward(self, x: torch.Tensor):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ffwd(x))
        return x


class MyTfModel(MySeq2SeqModel):
    """ A transformer is consisted of num_layer number of AFBlocks
    """
    def __init__(self, config: MyTfConfig):
        super().__init__(config.embed_dim)
        self.config = config
        self.af_blocks = nn.ModuleList([AFBlock(config) for _ in range(config.num_layers)])

    def forward(self, x: torch.Tensor,
                output_hidden_states: Optional[bool] = None):
        """ x : (seq_len, batch_size, embed_dim)
        """
        all_hidden_states = () if output_hidden_states else None

        z = x
        for block in self.af_blocks:
            z = block(z)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (z,)
                
        return ModelOutput(tensor = z,
                           all_hidden_states = all_hidden_states)


