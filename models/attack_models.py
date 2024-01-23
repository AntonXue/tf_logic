import copy
from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.utils import ModelOutput

from .common import *


""" Different attacker models """


@dataclass
class ForceOutputWithAppendedAttackTokensOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    norm_loss: Optional[torch.FloatTensor] = None
    pred_loss: Optional[torch.FloatTensor] = None
    attack_tokens: Optional[torch.FloatTensor] = None
    pred: Optional[torch.FloatTensor] = None


class ForceOutputWithAppendedAttackTokensWrapper(nn.Module):
    def __init__(
        self,
        seqcls_model: SeqClsModel,
        num_attack_tokens: int,
        base_attack_model_name: str = "gpt2",
        rho: float = 1e-3
    ):
        super().__init__()
        seqcls_model = copy.deepcopy(seqcls_model)
        for p in seqcls_model.parameters():
            p.requires_grad = False

        seqcls_model.eval()
        self.seqcls_model = seqcls_model
        self.input_dim = input_dim = seqcls_model.input_dim
        self.num_labels = num_labels = seqcls_model.num_labels
        self.num_attack_tokens = num_attack_tokens

        if base_attack_model_name == "gpt2":
            gpt2 = AutoModel.from_pretrained("gpt2")
            self.attack_model = gpt2
            self.embed_dim = embed_dim = gpt2.embed_dim
        else:
            self.embed_dim = embed_dim = 2048
            self.attack_model = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        self.tokens_embed_fn = nn.Linear(input_dim, embed_dim)
        self.target_embed_fn = nn.Linear(num_labels, embed_dim)
        self.attack_cls_head = nn.Linear(embed_dim, input_dim)

        self.rho = rho
        self.norm_loss_fn = nn.MSELoss(reduction="sum")
        self.pred_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    def train(self, mode=True):
        """ We don't want the seqcls_model under attack to be in training mode """
        self.seqcls_model.eval()
        self.attack_model.train(mode=mode)

    def forward(
        self,
        tokens: torch.LongTensor,
        target: torch.LongTensor,
    ):
        """ tokens: (batch_size, seq_len, input_dim)
            target: (batch_size, num_labels)
        """
        assert tokens.size(1) > self.num_attack_tokens

        atk_model_input = torch.cat([
            self.tokens_embed_fn(tokens.float()).view(-1, tokens.size(1), self.embed_dim),
            self.target_embed_fn(target.float()).view(-1, 1, self.embed_dim)
        ], dim=1)    # (batch_size, seq_len+1, embed_dim)

        z = self.attack_model(inputs_embeds=atk_model_input).last_hidden_state
        atk_tokens = self.attack_cls_head(z)[:,:self.num_attack_tokens]
        pred = self.seqcls_model(torch.cat([tokens, atk_tokens], dim=1)).logits

        norm_loss = self.rho * self.norm_loss_fn(torch.zeros_like(atk_tokens), atk_tokens)
        pred_loss = self.pred_loss_fn(pred, target.float())

        return ForceOutputWithAppendedAttackTokensOutput(
            loss = norm_loss + pred_loss,
            norm_loss = norm_loss,
            pred_loss = pred_loss,
            attack_tokens = atk_tokens,
            pred = pred
        )


