from typing import Optional, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from ..common import *

""" Our custom transformer model """

class MyTfConfig:
    """ Use this to initialize MyTfModel and its related classes.
        The following setup lets us conveniently initialize with Nones,
        which may come in by default from command line argmuents
    """
    def __init__(
        self,
        input_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        ffwd_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        do_layer_norm: Optional[bool] = None,
        num_labels: Optional[int] = None,
        problem_type: Optional[str] = None,
        use_positional_encoding: Optional[bool] = None,
        max_seq_len: Optional[int] = None,
    ):
        self.input_dim = default(input_dim, 256)
        self.embed_dim = default(embed_dim, 512)
        self.ffwd_dim = default(ffwd_dim, 4 * self.embed_dim)
        self.num_heads = default(num_heads, 4)
        self.num_layers = default(num_layers, 8)
        self.do_layer_norm = default(do_layer_norm, True)
        self.num_labels = default(num_labels, None)
        self.problem_type = default(problem_type, "multi_label_classification")
        self.use_positional_encoding = default(use_positional_encoding, False)
        self.max_seq_len = default(max_seq_len, 1024) if self.use_positional_encoding else None

    def to_dict(self):
        """ This is required by the HF Trainer """
        return {
            "model_name": "mytf"
        }


@dataclass
class MyTfOutput(ModelOutput):
    """ At the moment this is the same as BaseModelOutput, but may change """
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class MySelfAttention(nn.Module):
    def __init__(self, config: MyTfConfig):
        super().__init__()
        self.embed_dim = embed_dim = config.embed_dim
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.FloatTensor):
        """ x: (batch_size, seq_len, embed_dim) """
        N, L, d = x.shape
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        mask = torch.triu(torch.ones(L,L), diagonal=1).view(1,L,L).repeat(N,1,1).to(x.device) * -999
        wts = F.softmax((torch.bmm(Q, K.transpose(1,2)) / (d**0.5)) + mask, dim=2) # (N,L,L)
        out = torch.bmm(wts, V)
        return out, wts


class MyAFBlock(nn.Module):
    """ A single attention-feedforward block, simplified """
    def __init__(self, config: MyTfConfig):
        super().__init__()
        # Norm layers
        if config.do_layer_norm:
            self.norm1 = nn.LayerNorm(config.embed_dim)
            self.norm2 = nn.LayerNorm(config.embed_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Attention block
        assert config.num_heads > 0
        self.attn_heads = nn.ModuleList([MySelfAttention(config) for _ in range(config.num_heads)])

        # Feedforward block construction
        self.ffwd = nn.Sequential(
            nn.Linear(config.embed_dim, config.ffwd_dim),
            nn.ReLU(),
            nn.Linear(config.ffwd_dim, config.embed_dim)
        )

    def forward(self, x: torch.FloatTensor):
        """ x: (batch_size, seq_len, embed_dim) """
        # Apply attentions
        z = self.norm1(x)
        attn_out, attn_wts = zip(*[ah(z) for ah in self.attn_heads])
        attn_wts = torch.stack(attn_wts, dim=1)
        x = x + sum(attn_out)

        # Apply the feedforward
        x = x + self.ffwd(self.norm2(x))
        return x, attn_wts


class MyTfModel(nn.Module):
    """ A transformer is made of num_layer number of AFBlocks, maps embed_dim to embed_dim
    """
    def __init__(self, config: MyTfConfig):
        super().__init__()
        self.config = config
        self.af_blocks = nn.ModuleList([MyAFBlock(config) for _ in range(config.num_layers)])
        if config.use_positional_encoding:
            self.positional_encoding = nn.Embedding(config.max_seq_len, config.embed_dim)

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
        if self.config.use_positional_encoding:
            x = x + self.positional_encoding(torch.arange(0, x.size(1)).to(x.device))

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
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class MyTfSeqClsModel(SeqClsModel):
    """ Maps input_dim to num_labels """
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
        self.embed_fn = nn.Linear(config.input_dim, config.embed_dim)
        self.cls_head = nn.Linear(config.embed_dim, config.num_labels)

    @property
    def model_name(self):
        return "mytf"

    @property
    def input_dim(self):
        return self.config.input_dim

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
        """ x : (batch_size, seq_len, embed_dim) """
        x = self.embed_fn(x)

        tf_out = self.mytf(
            x,
            output_hidden_states = output_hidden_states,
            output_attentions = output_attentions
        )

        logits = self.cls_head(tf_out.last_hidden_state)[:,-1]   # (batch_size, num_labels)

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
            attentions = tf_out.attentions
        )


