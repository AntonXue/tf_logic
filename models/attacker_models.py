from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModel

from .common import *

""" Different attacker models """

class AttackerModel(nn.Module):
    def __init__(self):
        super().__init__()


class ForceOutputAttackerModel(AttackerModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_outputs: int,
        seq2seq_model_name: str = "gpt2"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_outputs = num_outputs

        if seq2seq_model_name == "gpt2":
            gpt2 = AutoModel.from_pretrained("gpt2")
            self.seq2seq_model = gpt2
            self.embed_dim = embed_dim = gpt2.embed_dim
        else:
            self.embed_dim = embed_dim = 2048
            self.seq2seq_model = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim))

        self.tokens_embed_fn = nn.Linear(input_dim, embed_dim)
        self.target_embed_fn = nn.Linear(output_dim, embed_dim)
        self.cls_head = nn.Linear(embed_dim, output_dim)

    def forward(
        self,
        tokens: torch.FloatTensor,
        target: torch.FloatTensor,
    ):
        """ tokens: (batch_size, seq_len, input_dim)
            target: (batch_size, output_dim)
        """

        x = torch.cat([
            self.tokens_embed_fn(tokens.float()).view(-1, tokens.size(1), self.embed_dim),
            self.target_embed_fn(target.float()).view(-1, 1, self.embed_dim)
        ], dim=1)    # (batch_size, seq_len+1, embed_dim)


        print(f"x: {x.shape}")
        z = self.seq2seq_model(inputs_embeds=x).last_hidden_state
        out = self.cls_head(z)  # (batch_size, seq_len+1, output_dim)
        return out[:,:self.num_outputs] # get the first few elements


