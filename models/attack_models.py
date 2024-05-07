import copy
from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
# from transformers import GPT2Config, GPT2Model
from transformers import GPT2ForSequenceClassification

from .common import *
from .task_models import AutoregKStepsTaskModel


""" Different attacker models """

@dataclass
class AttackerModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    others: Optional[dict] = None


class CoerceStateWrapperModel(nn.Module):
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
        elif self.token_range == "clamped":
            atk_tokens = torch.cat([torch.zeros_like(atk_logits), atk_logits], dim=-1).clamp(0,1)

        # Adversarial (and its binary version) to the reasoner model and query it
        adv_inputs = torch.cat([atk_tokens, tokens], dim=1)
        out = self.reasoner_model(adv_inputs)
        logits = out.logits[:,0] # The first item of the autoreg sequence

        loss = nn.BCEWithLogitsLoss()(logits, labels.float())

        return AttackerModelOutput(
            loss = loss,
            logits = logits,
        )


class SuppressRuleWrapperModel(nn.Module):
    def __init__(
        self,
        reasoner_model: AutoregKStepsTaskModel,
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

        # Special thing here
        self.token_embed_fn = nn.Linear(self.token_dim, self.embed_dim)
        self.rule_embed_fn = nn.Linear(self.token_dim, self.embed_dim)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def train(self, mode=True):
        """ We don't want the seqcls_model under attack to be in training mode """
        self.reasoner_model.eval()
        self.attacker_model.train(mode=mode)


    def forward(
        self,
        tokens: torch.LongTensor,
        abcde: torch.LongTensor,
        labels: torch.LongTensor,
    ):
        N, L, _ = tokens.shape
        tokens, labels = tokens.float(), labels.float()
        device = tokens.device

        # Our goal is to suppress the rule (a,b)
        a, b, c, d, e = abcde.chunk(5, dim=-1)
        n = self.num_vars

        supp_ante = F.one_hot(a, num_classes=n).view(-1,1,n)
        supp_conseq = F.one_hot(b, num_classes=n).view(-1,1,n)
        supp_rule = torch.cat([supp_ante, supp_conseq], dim=-1).float()

        # Query the attacker model
        atk_inputs_embeds = torch.cat([
            self.rule_embed_fn(supp_rule).view(N,1,-1),
            self.token_embed_fn(tokens).view(N,L,-1),
        ], dim=1)

        atk_logits = self.attacker_model(inputs_embeds=atk_inputs_embeds).logits

        # Prepend the logits to the token sequence
        adv_inputs = torch.cat([atk_logits.view(-1,1,self.token_dim), tokens], dim=1)
        res_out = self.reasoner_model(tokens=adv_inputs)
        res_logits = res_out.logits #

        # Amplify the component regarding b

        loss = None
        if labels is not None:
            loss = self.loss_fn(res_logits, labels.float())

        return AttackerModelOutput(
            loss = loss,
            logits = atk_logits,
            others = {
                "reasoner_output": res_out,
            }
        )


