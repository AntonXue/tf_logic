from typing import Optional, List
from dataclasses import dataclass
import copy
import math
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from .common import *

""" Our custom transformer model """

@dataclass
class MyTfConfig:
    """ Use this to initialize MyTfModel """
    embed_dim: int
    ffwd_width: int
    ffwd_depth: int
    num_heads: int
    num_layers: int
    activ: str = "relu"
    do_norm: bool = True
    layer_norm_epsilon: float = 1e-5


@dataclass
class MyTfOutput(ModelOutput):
    """ At the moment this is the same as BaseModelOutput, but may change """
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attentions: Optional[List[torch.FloatTensor]] = None


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
        ffwd_parts = [nn.Linear(config.embed_dim, config.ffwd_width), get_activ(config.activ)]
        for i in range(config.ffwd_depth-1):
            ffwd_parts += [nn.Linear(config.ffwd_width, config.ffwd_width), get_activ(config.activ)]
        ffwd_parts.append(nn.Linear(config.ffwd_width, config.embed_dim))
        self.ffwd = nn.Sequential(*ffwd_parts)

    def forward(self, x: torch.FloatTensor):
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
            x: torch.FloatTensor,
            output_hidden_states: Optional[bool] = None,
            output_attentions : Optional[bool] = None):
        """ x : (batch_size, seq_len, embed_dim) """
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for block in self.af_blocks:
            x, a = block(x)

            if output_hidden_states:
                all_hidden_states.append(x)

            if output_attentions:
                all_attentions.append(a)

        return MyTfOutput(
            last_hidden_state = x,
            hidden_states = all_hidden_states,
            attentions = all_attentions)


""" Other modules built on top of the basic transformer """

@dataclass
class MyTfForSeqClsConfig:
    mytf_config: MyTfConfig
    num_labels: int

    def __post_init__(self):
        for k, v in asdict(self.mytf_config).items():
            self.__setattr__(k, v)


@dataclass
class MyTfForSeqClsOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attentions: Optional[List[torch.FloatTensor]] = None


class MyTfForSeqCls(MySeqClsModel):
    def __init__(self, config: MyTfForSeqClsConfig):
        super().__init__(config.embed_dim, num_labels=config.num_labels, max_seq_len=None)
        self.config = config
        self.mytf = MyTfModel(config)
        self.num_labels = config.num_labels
        self.cls_head = nn.Linear(config.embed_dim, config.num_labels)

    def forward(
            self,
            x: torch.FloatTensor,
            output_hidden_states: Optional[bool] = None,
            output_attentions : Optional[bool] = None):
        tf_out = self.mytf(x, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        logits = self.cls_head(tf_out.last_hidden_state)[:,0]
        return MyTfForSeqClsOutput(
                logits = logits,
                last_hidden_state = tf_out.last_hidden_state,
                hidden_states = tf_out.hidden_states,
                attentions = tf_out.attentions)


