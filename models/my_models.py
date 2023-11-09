from typing import Optional, List, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from .common import *

""" Our custom transformer model """

class MyTfConfig:
    """ Use this to initialize MyTfModel and its related classes """
    def __init__(
            self,
            embed_dim: int = 512,
            ffwd_width: int = 1024,
            ffwd_depth: int = 4,
            num_heads: int = 4,
            num_layers: int = 8,
            activ: str = "relu",
            do_norm: bool = True,
            layer_norm_epsilon: float = 1e-5,
            num_labels: Optional[int] = None,
            problem_type: Optional[str] = None,
            **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffwd_width = ffwd_width
        self.ffwd_depth = ffwd_depth
        self.activ = activ
        self.do_norm = do_norm
        self.layer_norm_epsilon = layer_norm_epsilon
        self.num_labels = num_labels
        self.problem_type = problem_type

        for k, v in kwargs.items():
            assert not hasattr(self, k) # Make sure to not overwrite something
            self.__setattr__(k, v)


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


class MyTfModel(nn.Module):
    """ A transformer is consisted of num_layer number of AFBlocks """
    def __init__(self, config: MyTfConfig):
        super().__init__()
        self.config = config
        self.af_blocks = nn.ModuleList([AFBlock(config) for _ in range(config.num_layers)])

    @property
    def embed_dim(self):
        return config.embed_dim

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
class MyTfSeqClsOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attentions: Optional[List[torch.FloatTensor]] = None


class MyTfSeqClsModel(nn.Module):
    def __init__(self, config: MyTfConfig):
        super().__init__()

        if config.problem_type == "regression":
            assert config.num_labels >= 1
        elif config.problem_type in ["single_label_classification", "multi_label_classification"]:
            assert config.num_labels > 1
        else:
            raise ValueError(f"Bad problem type {self.config.problem_type}")

        self.config = config
        self.mytf = MyTfModel(config)
        self.encoder = nn.Linear(config.embed_dim, config.embed_dim)
        self.cls_head = nn.Linear(config.embed_dim, config.num_labels)

    @property
    def embed_dim(self):
        return self.config.embed_dim

    @property
    def num_labels(self):
        return self.config.num_labels

    def forward(
            self,
            x: torch.FloatTensor,
            output_hidden_states: Optional[bool] = None,
            output_attentions : Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None):
        x = self.encoder(x)
        tf_out = self.mytf(x, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        logits = self.cls_head(tf_out.last_hidden_state)[:,0]   # (batch_size, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type == "regression":
                loss = nn.MSELoss()(logits.squeeze(), labels.view(-1))
            elif self.config.problem_type == "single_label_classification":
                loss = nn.CrossEntropyLoss()(logits, labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            else:
                raise NotImplementedError()

        return MyTfSeqClsOutput(
                loss = loss,
                logits = logits,
                last_hidden_state = tf_out.last_hidden_state,
                hidden_states = tf_out.hidden_states,
                attentions = tf_out.attentions)


