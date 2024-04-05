from dataclasses import dataclass
from typing import Optional, Union
import torch
import transformers

from transformers import GPT2Config, GPT2ForSequenceClassification

from ..common import *

@dataclass
class MyGPT2Config:
    input_dim: Optional[int] = None
    num_vars: Optional[int] = None
    embed_dim: Optional[int] = None
    ffwd_dim: Optional[int] = None
    num_heads: Optional[int] = 1
    num_layers: Optional[int] = 1
    problem_type: str = "multi_label_classification"
    max_seq_len: Optional[int] = 1024
    use_positional_embedding: bool = False
    use_pretrained: Optional[bool] = False # This will override everything except num_vars
    pad_token_id: Optional[int] = -1 # Maybe change to something smarter


class MyGPT2SeqClsModel(SeqClsModel):
    def __init__(self, config: MyGPT2Config):
        super().__init__()
        self.config = config

        if config.use_pretrained:
            self.gpt2s = GPT2ForSequenceClassification.from_pretrained(
                "gpt2",
                ignore_mismatched_size = True,
                num_labels = config.num_vars,
                problem_type = config.problem_type,
                pad_token_id = config.pad_token_id,
            )

        else:
            self.gpt2s = GPT2ForSequenceClassification._from_config(GPT2Config(
                n_embd = config.embed_dim,
                n_inner = config.ffwd_dim,
                n_head = config.num_heads,
                n_layer = config.num_layers,
                num_labels = config.num_vars,
                problem_type = config.problem_type,
                max_position_embeddings = config.max_seq_len,
                pad_token_id = config.pad_token_id
            ))

        self.model_config = self.gpt2s.config

        if config.input_dim is not None:
            self.embed_fn = nn.Linear(config.input_dim, config.embed_dim)

    @property
    def model_name(self):
        return "gpt2"

    @property
    def input_dim(self):
        return self.config.input_dim

    @property
    def embed_dim(self):
        return self.config.embed_dim

    @property
    def num_labels(self):
        return self.config.num_vars

    @property
    def propblem_type(self):
        return self.config.problem_type

    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
            x:         (batch_size, seq_len, input_dim)
            input_ids: (batch_size, seq_len)
        """
        assert not (x is None and input_ids is None)
        assert not (x is not None and input_ids is not None)

        # Use inputs_embeds
        if input_ids is None:
            x = self.embed_fn(x)

            if self.config.problem_type == "multi_label_classification" and labels is not None:
                labels = labels.float()

            out = self.gpt2s(
                inputs_embeds = x,
                attention_mask = attention_mask,
                labels = labels,
                **kwargs
            )

            return out

        # Use input_ids
        else:
            out = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels,
                **kwargs
            )

            return out
            

