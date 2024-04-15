import copy
from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
# from transformers import GPT2Config, GPT2Model
from transformers import GPT2ForSequenceClassification

from .common import *


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
        self.token_dim = 2 * self.num_vars
        self.attacker_model = GPT2ForSequenceClassification.from_pretrained(
            "gpt2",
            num_labels = num_attack_tokens * self.token_dim,
            problem_type = "multi_label_classification",
            pad_token_id = -1
        )
        self.embed_dim = self.attacker_model.transformer.embed_dim

        self.tokens_embed_fn = nn.Linear(self.token_dim, self.embed_dim)
        self.target_embed_fn = nn.Linear(self.num_vars, self.embed_dim)


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
                self.target_embed_fn(labels).view(N,1,-1),
            ], dim=1
        )

        # z = self.attacker_model(inputs_embeds=atk_inputs_embeds).last_hidden_state
        # atk_logits = self.cls_head(z)[:,:self.num_attack_tokens]
        atk_logits = self.attacker_model(inputs_embeds=atk_inputs_embeds).logits
        atk_logits = atk_logits.view(-1, self.num_attack_tokens, self.token_dim)

        # Depending on the token mode, clamp if necessary and binarize accordingly
        if self.token_range == "unbounded":
            atk_tokens = atk_logits
            bin_atk_tokens = (atk_tokens > 0.0).long()
        elif self.token_range == "clamped":
            atk_tokens = (atk_logits*0.5 + 0.5).clamp(0,1)
            bin_atk_tokens = (atk_tokens > 0.5).long()

        # Adversarial (and its binary version) to the reasoner model and query it
        adv_inputs = torch.cat([tokens, atk_tokens], dim=1)
        res_out = self.reasoner_model(adv_inputs)
        # res_logits = res_out.logits[:,0] # The first item of the autoreg sequence
        res_logits = res_out.logits[:,-1] # Try the last logit

        with torch.no_grad():
            # Don't apply grad here because binarization as we wrote is not differentiable
            bin_adv_inputs = torch.cat([tokens, bin_atk_tokens], dim=1)
            bin_res_out = self.reasoner_model(bin_adv_inputs)
            bin_res_logits = bin_res_out.logits[:,0]

        loss = nn.BCEWithLogitsLoss()(res_logits, labels.float())
        # loss += 1e-3 * nn.MSELoss()(atk_tokens, torch.zeros_like(atk_tokens))
        return AttackerModelOutput(
            loss = loss,
            logits = torch.stack([res_logits, bin_res_logits], dim=1), # (N,2,n),
            attack_tokens = atk_tokens,
            bin_attack_tokens = bin_atk_tokens
        )


