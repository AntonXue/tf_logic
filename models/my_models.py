from typing import Optional
from dataclasses import dataclass

import copy
import math
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from .utils import *


""" Our custom transformer model """

@dataclass
class MyTfConfig:
    embed_dim: int
    ffwd_width: int
    ffwd_depth: int
    num_heads: int
    num_layers: int
    ffwd_activ: str = "relu"
    do_norm: bool = True
    layer_norm_epsilon: float = 1e-5


class AFBlock(nn.Module):
    """ A single attention-feedforward block """
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
        z, a = self.attn(x, x, x)
        z = self.norm1(x + z)
        z = self.norm2(z + self.ffwd(z))
        return z, a


class MyTfModel(MySeq2SeqModel):
    """ A transformer is consisted of num_layer number of AFBlocks """
    def __init__(self, config: MyTfConfig):
        super().__init__(config.embed_dim, max_seq_len=None)
        self.config = config
        self.af_blocks = nn.ModuleList([AFBlock(config) for _ in range(config.num_layers)])

    def forward(
            self,
            x: torch.Tensor,
            output_hidden_states: Optional[bool] = None,
            output_attentions : Optional[bool] = None):
        """ x : (batch_size, seq_len, embed_dim) """

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block in self.af_blocks:
            x, a = block(x)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

            if output_attentions:
                all_attentions = all_attentions + (a,)

        # Will have to change from BaseModelOutput if we want to add more keys
        return BaseModelOutput(
            last_hidden_state = x,
            hidden_states = all_hidden_states,
            attentions = all_attentions
        )


