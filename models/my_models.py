from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from .common import *

""" Our custom transformer model """

class MyTfConfig:
    """ Use this to initialize MyTfModel and its related classes.
        The following setup lets us conveniently initialize with Nones.
    """
    def __init__(
        self,
        embed_dim: Optional[int] = None,
        ffwd_width: Optional[int] = None,
        ffwd_depth: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        activ: Optional[str] = None,
        do_norm: Optional[bool] = None,
        layer_norm_epsilon: Optional[float] = None,
        num_labels: Optional[int] = None,
        problem_type: Optional[str] = None,
        max_seq_len: Optional[int] = None
    ):
        self.embed_dim = default(embed_dim, 512)
        self.ffwd_width = default(ffwd_width, 1024)
        self.ffwd_depth = default(ffwd_depth, 4)
        self.num_heads = default(num_heads, 4)
        self.num_layers = default(num_layers, 8)
        self.activ = default(activ, "relu")
        self.do_norm = default(do_norm, True)
        self.layer_norm_epsilon = default(layer_norm_epsilon, 1e-5)
        self.num_labels = default(num_labels, None)
        self.problem_type = default(problem_type, None)
        self.max_seq_len = default(max_seq_len, 1024)


@dataclass
class MyTfOutput(ModelOutput):
    """ At the moment this is the same as BaseModelOutput, but may change """
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


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
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

    @property
    def model_name(self):
        return "mytf"

    @property
    def embed_dim(self):
        return self.config.embed_dim

    def forward(
        self,
        x: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions : Optional[bool] = None
    ):
        """ x : (batch_size, seq_len, embed_dim) """
        x = x + self.pos_embedding(torch.arange(0, x.size(1)).to(x.device))

        all_hidden_states = (x,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block in self.af_blocks:
            x, a = block(x)

            if output_hidden_states:
                all_hidden_states += (x,)

            if output_attentions:
                all_attentions += (a,)

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
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MyTfSeqClsModel(nn.Module):
    def __init__(self, config: MyTfConfig):
        super().__init__()

        if config.problem_type == "regression":
            assert config.num_labels >= 1
        elif config.problem_type == "single_label_classification":
            assert config.num_labels > 1
        elif config.problem_type == "multi_label_classification":
            assert config.num_labels > 1
        else:
            raise ValueError(f"Bad problem type {config.problem_type}")

        self.config = config
        self.mytf = MyTfModel(config)
        self.encoder = nn.Linear(config.embed_dim, config.embed_dim)
        self.cls_head = nn.Linear(config.embed_dim, config.num_labels)

    @property
    def model_name(self):
        return "mytf"

    @property
    def embed_dim(self):
        return self.config.embed_dim

    @property
    def num_labels(self):
        return self.config.num_labels

    @property
    def problem_type(self):
        return self.config.problem_type

    def forward(
        self,
        x: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions : Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None
    ):
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


