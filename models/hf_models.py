from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import transformers
from transformers import GPT2Model, GPT2Config

from .model_utils import *

""" Some models from Hugging Face
"""

@dataclass
class MyGPT2Config:
    use_pretrained: bool = False
    pretrained_id: Optional[str] = None
    gpt2_config_kwargs: Optional[dict] = None
    gpt2_config: Optional[GPT2Config] = None

    def __post_init__(self):
        if self.use_pretrained:
            # Can't have both pretrained and kwargs
            assert self.gpt2_config_kwargs is None and self.gpt2_config is None
            self.pretrained_id = default(self.pretrained_id, "gpt2")
            self.__setattr__("embed_dim", 768)
        else:
            if self.gpt2_config is None:
                config_kwargs = default(self.gpt2_config_kwargs, {})
                self.gpt2_config = GPT2Config(**config_kwargs)
            self.__setattr__("embed_dim", self.gpt2_config.n_embd)


class MyGPT2Model(MySeq2SeqModel):
    """ Wrap the GPT2 model from Hugging Face
    """
    def __init__(self, config: MyGPT2Config):
        super().__init__(config.embed_dim)
        self.config = config

        if config.use_pretrained:
            self.gpt2 = GPT2Model.from_pretrained(config.pretrained_id)
            self.config.__setattr__("gpt2_config", self.gpt2.config)
        else:
            self.gpt2 = GPT2Model(config.gpt2_config)


    def forward(self, x: torch.Tensor, **kwargs):
        """ x : (seq_len, batch_size, embed_dim)
        """
        out = self.gpt2(inputs_embeds=x, **kwargs)
        out.__setitem__("tensor", out["last_hidden_state"])
        return out


