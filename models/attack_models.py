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
    others: Optional[dict] = None


class BadSuffixWrapperModel(nn.Module):
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
            num_labels = num_attack_tokens * self.num_vars,
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

        atk_logits = self.attacker_model(inputs_embeds=atk_inputs_embeds).logits
        atk_logits = atk_logits.view(-1, self.num_attack_tokens, self.num_vars)

        # Depending on the token mode, clamp if necessary and binarize accordingly
        if self.token_range == "unbounded":
            atk_tokens = torch.cat([torch.zeros_like(atk_logits), atk_logits], dim=-1)
            bin_atk_tokens = (atk_tokens > 0.0).long()
        elif self.token_range == "clamped":
            atk_tokens = torch.cat([torch.zeros_like(atk_logits), atk_logits], dim=-1).clamp(0,1)
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

        logits = torch.cat([
            atk_logits,
            torch.stack([res_logits, bin_res_logits], dim=1)
        ], dim=1) # (N,k+2,n)

        return AttackerModelOutput(
            loss = loss,
            logits = logits,
        )


class SuppressRuleWrapperModel(nn.Module):
    def __init__(
        self,
        reasoner_model: SeqClsModel,
    ):
        super().__init__()

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
            num_labels = self.token_dim,
            problem_type = "multi_label_classification",
            pad_token_id = -1
        )
        self.embed_dim = self.attacker_model.transformer.embed_dim
        self.token_embed_fn = nn.Linear(self.token_dim, self.embed_dim)
        self.rule_embed_fn = nn.Linear(self.token_dim, self.embed_dim)


    def train(self, mode=True):
        """ We don't want the seqcls_model under attack to be in training mode """
        self.reasoner_model.eval()
        self.attacker_model.train(mode=mode)


    def forward(
        self,
        tokens: torch.LongTensor,
        supp_rule: torch.LongTensor,
        labels,
        **kwargs
    ):
        N, L, _ = tokens.shape
        tokens, supp_rule = tokens.float(), supp_rule.float()

        # Query the attacker model
        atk_inputs_embeds = torch.cat([
                self.token_embed_fn(tokens).view(N,L,-1),
                self.rule_embed_fn(supp_rule).view(N,1,-1)
            ], dim=1
        )

        atk_logits = self.attacker_model(inputs_embeds=atk_inputs_embeds).logits

        # Prepend the logits to the token sequence
        adv_inputs = torch.cat([atk_logits.view(-1,1,self.token_dim), tokens], dim=1)
        res_out = self.reasoner_model(adv_inputs)
        res_logits = res_out.logits[:,0]    # The first item of the autoreg sequence

        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(res_logits, labels.float())

        supp_ante, supp_conseq = supp_rule.chunk(2, dim=-1)
        atk_ante, atk_conseq = atk_logits.chunk(2, dim=-1)

        logits = torch.stack([
                supp_ante,
                supp_conseq,
                atk_ante,
                atk_conseq,
                res_logits
            ], dim=1
        )

        return AttackerModelOutput(
            loss = loss,
            logits = logits,
        )


