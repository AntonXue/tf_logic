import copy
from typing import Optional
from dataclasses import dataclass
from models.__init__ import AutoSeqClsModel
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from transformers import GPT2Model

from .common import *
# from models import AutoTaskModel, AutoSeqClsModel


""" Different attacker models """

@dataclass
class AttackerModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    attack_tokens: Optional[torch.FloatTensor] = None
    bin_attack_tokens: Optional[torch.LongTensor] = None


class AttackWrapperModel(nn.Module):
    def __init__(
        self,
        reasoner_model: SeqClsModel,
        num_attack_tokens: int,
        token_range: str = "unbounded",
    ):
        super().__init__()
        assert token_range in ["unbounded", "clamped"]
        self.token_range = token_range
        self.num_attack_tokens = num_attack_tokens

        # Set up the reasoner model
        reasoner_model = copy.deepcopy(reasoner_model)
        for p in reasoner_model.parameters():
            p.requires_grad = False
        reasoner_model.eval()
        self.reasoner_model = reasoner_model

        self.num_vars = reasoner_model.num_labels
        self.attacker_model = GPT2Model.from_pretrained("gpt2")
        self.embed_dim = self.attacker_model.embed_dim

        self.tokens_embed_fn = nn.Linear(2 * self.num_vars, self.embed_dim)
        self.labels_embed_fn = nn.Linear(self.num_vars, self.embed_dim)
        self.cls_head = nn.Linear(self.embed_dim, 2 * self.num_vars)


    def train(self, mode=True):
        """ We don't want the seqcls_model under attack to be in training mode """
        self.reasoner_model.eval()
        self.attacker_model.train(mode=mode)

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: torch.LongTensor,   # The adversarial target
    ):

        N, L, _ = tokens.shape
        tokens, labels = tokens.float(), labels.float()

        # Query the attacker model
        atk_inputs_embeds = torch.cat([
                self.tokens_embed_fn(tokens).view(N,L,-1),
                self.labels_embed_fn(labels).view(N,1,-1),
            ], dim=1
        )

        atk_hidden_state = self.attacker_model(inputs_embeds=atk_inputs_embeds).last_hidden_state
        atk_logits = self.cls_head(atk_hidden_state)[:,:self.num_attack_tokens]

        # Depending on the token mode, clamp if necessary and binarize accordingly
        if self.token_range == "clamped":
            atk_tokens = (atk_logits*0.5 + 0.5).clamp(0,1)
            bin_atk_tokens = (atk_tokens > 0.5).long()
        else:
            atk_tokens = atk_logits
            bin_atk_tokens = (atk_tokens > 0.0).long()

        # Adversarial (and its binary version) to the reasoner model and query it
        adv_inputs = torch.cat([tokens, atk_tokens], dim=1)
        res_out = self.reasoner_model(adv_inputs)
        res_logits = res_out.logits[:,0] # The first item of the autoreg sequence

        with torch.no_grad():
            # Don't apply grad here because binarization as we wrote is not differentiable
            bin_adv_inputs = torch.cat([tokens, bin_atk_tokens], dim=1)
            bin_res_out = self.reasoner_model(bin_adv_inputs)
            bin_res_logits = bin_res_out.logits[:,0]

        loss = nn.BCEWithLogitsLoss()(res_logits, labels.float())
        return AttackerModelOutput(
            loss = loss,
            logits = torch.stack([res_logits, bin_res_logits], dim=1), # (N,2,n),
            attack_tokens = atk_tokens,
            bin_attack_tokens = bin_atk_tokens
        )


