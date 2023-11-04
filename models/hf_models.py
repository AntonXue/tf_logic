from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import transformers
from transformers import GPT2Config, GPT2Model, GPT2ForSequenceClassification

from .common import *

""" Some models from Hugging Face """

class MyGPT2Config:
    def __init__(
            self,
            use_pretrained: bool = False,
            pretrained_checkpoint: Optional[str] = None,
            max_seq_len: Optional[int] = None,
            **kwargs):

        self.use_pretrained = use_pretrained
        if use_pretrained:
            self.pretrained_checkpoint = pretrained_checkpoint
            self.pretrained_kwargs = kwargs

        else:
            self.gpt2_config_kwargs = kwargs
            self.gpt2_config = GPT2Config(**self.gpt2_config_kwargs)


class MyGPT2Model(MySeq2SeqModel):
    """ Wrap the GPT2 model from Hugging Face """
    def __init__(self, config: MyGPT2Config):
        if config.use_pretrained:
            gpt2 = GPT2Model.from_pretrained(config.pretrained_checkpoint, **config.pretrained_kwargs)
        else:
            gpt2 = GPT2Model(config.gpt2_config)

        super().__init__(gpt2.config.n_embd, gpt2.config.n_positions)
        self.config = config
        self.gpt2 = gpt2

    def forward(self, x: torch.Tensor, use_input_ids: Optional[bool] = None, **kwargs):
        """ x : (batch_size, seq_len, embed_dim) """
        if use_input_ids:
            out = self.gpt2(input_ids=x, **kwargs)
        else:
            out = self.gpt2(inputs_embeds=x, **kwargs)
        return out  # This is a BaseModelOutputWithPastAndCrossAttentions


class MyGPT2ForSeqCls(MySeqClsModel):
    """ Wrap the GPT2ForSequenceClassification Model from Hugging Face """
    def __init__(self, config: MyGPT2Config):
        if config.use_pretrained:
            assert hasattr(config, "pretrained_kwargs") and hasattr(config.pretrained_kwargs, "num_labels")
            gpt2 = GPT2ForSequenceClassification.from_pretrained(
                    config.pretrained_checkpoint, **config.pretrained_kwargs)
        else:
            assert hasattr(config.gpt2_config, "problem_type") and hasattr(config.gpt2_config, "num_labels")
            gpt2 = GPT2ForSequenceClassification(config.gpt2_config)

        super().__init__(gpt2.config.n_embd, gpt2.config.num_labels,
                         gpt2.config.problem_type, gpt2.config.n_positions)
        self.config = config
        self.gpt2 = gpt2

    def forward(self, x: torch.Tensor, use_input_ids: Optional[bool] = None, **kwargs):
        """ x : (batch_size, seq_len, embed_dim) """
        if use_input_ids:
            out = self.gpt2(input_ids=x, **kwargs)
        else:
            out = self.gpt2(inputs_embeds=x, **kwargs)
        return out


