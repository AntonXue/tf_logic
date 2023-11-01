from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import transformers
from transformers import GPT2Model, GPT2Config

from .utils import *

""" Some models from Hugging Face """

@dataclass
class MyGPT2Config:
    use_pretrained: bool = False
    pretrained_checkpoint: Optional[str] = None
    gpt2_config: Optional[GPT2Config] = None
    gpt2_config_kwargs: Optional[dict] = None
    max_seq_len: Optional[int] = None

    def __post_init__(self):
        if self.use_pretrained:
            # If we're using pretrained stuff, can't have these
            assert self.gpt2_config is None \
                    and self.gpt2_config_kwargs is None \
                    and self.max_seq_len is None
            self.pretrained_checkpoint = default(self.pretrained_checkpoint, "gpt2")
        else:
            if self.gpt2_config is None:
                self.gpt2_config = GPT2Config()

            if self.max_seq_len is not None:
                self.gpt2_config.__setattr__("n_positions", self.max_seq_len)

            if self.gpt2_config_kwargs is not None:
                for k, v in self.gpt2_config_kwargs.items():
                    self.gpt2_config.__setattr__(k, v)


class MyGPT2Model(MySeq2SeqModel):
    """ Wrap the GPT2 model from Hugging Face """
    def __init__(self, config: MyGPT2Config):
        if config.use_pretrained:
            gpt2 = GPT2Model.from_pretrained(config.pretrained_checkpoint)
        else:
            gpt2 = GPT2Model(config.gpt2_config)

        super().__init__(gpt2.config.n_embd, gpt2.config.n_positions)
        self.config = config
        self.gpt2 = gpt2

    def forward(self, x: torch.Tensor, **kwargs):
        """ x : (batch_size, seq_len, embed_dim) """
        out = self.gpt2(inputs_embeds=x, **kwargs)
        # GPT2Model outputs a BaseModelOutputWithPastAndCrossAttentions
        return out


