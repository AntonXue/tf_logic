from typing import Optional
import torch
import torch.nn as nn
import transformers
from transformers import GPT2Model, GPT2Config

from .model_utils import *

""" Some models from Hugging Face
"""

class MyGPT2Config:
    """ Softly wrap the GPT2Config
    """
    def __init__(self,
                 from_pretrained: Optional[bool] = None,
                 pretrained_model_id: str = "gpt2",
                 **kwargs):
        self.from_pretrained = from_pretrained
        self.pretrained_model_id = pretrained_model_id
        self.gpt2_config = GPT2Config(**kwargs)

        if from_pretrained:
            self.embed_dim = 768
        else:
            self.embed_dim = self.gpt2_config.n_embd


class MyGPT2Model(MySeq2SeqModel):
    """ Wrap the GPT2 model from Hugging Face
    """
    def __init__(self, config: MyGPT2Config):
        super().__init__(config.embed_dim)
        self.config = config

        if config.from_pretrained:
            self.gpt2 = GPT2Model.from_pretrained(config.pretrained_model_id)
        else:
            self.gpt2 = GPT2Model(config.gpt2_config)


    def forward(self, x: torch.Tensor, **kwargs):
        """ x : (seq_len, batch_size, embed_dim)
        """
        out = self.gpt2(inputs_embeds=x, **kwargs)
        out.__setitem__("tensor", out["last_hidden_state"])
        return out


