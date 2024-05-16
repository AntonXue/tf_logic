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


class AttackWrapperModel(nn.Module):
    def __init__(
        self,
        reasoner_model: AutoregKStepsTaskModel,
        attack_name: str,
        num_attack_tokens: int = 1,
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

        # Set up the attacker model
        assert attack_name in ["suppress_rule", "knowledge_amnesia", "coerce_state"]
        self.attack_name = attack_name
        self.num_attack_tokens = num_attack_tokens
        self.attacker_model = GPT2ForSequenceClassification.from_pretrained(
            "gpt2",
            num_labels = self.num_attack_tokens * self.token_dim,
            problem_type = "multi_label_classification",
            pad_token_id = -1
        )
        self.atk_embed_dim = self.attacker_model.transformer.embed_dim

        # Special thing here
        self.tokens_embed_fn = nn.Linear(self.token_dim, self.atk_embed_dim)
        self.hint_embed_fn = nn.Linear(self.num_vars, self.atk_embed_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(self, mode=True):
        """ We don't want the seqcls_model under attack to be in training mode """
        self.reasoner_model.eval()
        self.attacker_model.train(mode=mode)

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        target: Optional[None] = None,
        infos: Optional[torch.LongTensor] = None,
    ):
        N, L, _ = tokens.shape
        n = self.num_vars

        if self.attack_name == "suppress_rule":
            # Our goal is to suppress the rule (a,b)
            a, b = infos[:,0], infos[:,1]
            hint = F.one_hot(a,n) - F.one_hot(b,n)

        elif self.attack_name == "knowledge_amnesia":
            # Our goal is to suppress the rule (a,b)
            a = infos[:,0]
            hint = F.one_hot(a,n)

        elif self.attack_name == "coerce_state":
            hint = target

        else:
            raise ValueError(f"Unknown attack_name {self.attack_name}")

        # Query the attacker model
        hint, tokens = hint.float(), tokens.float()
        atk_inputs_embeds = torch.cat([
            self.hint_embed_fn(hint).view(N,1,-1),
            self.tokens_embed_fn(tokens).view(N,L,-1),
        ], dim=1)

        atk_logits = self.attacker_model(inputs_embeds=atk_inputs_embeds).logits
        atk_logits = atk_logits.view(N, self.num_attack_tokens, self.token_dim)

        adv_tokens = torch.cat([tokens, atk_logits], dim=1)
        res_out = self.reasoner_model(tokens=adv_tokens)
        res_logits = res_out.logits

        loss = None
        if labels is not None:
            # We need to also factor in the initial state's logits
            all_logits = torch.cat([
                atk_logits[:,-1,self.num_vars:].view(N,1,self.num_vars),
                res_logits
            ], dim=1)

            loss = self.loss_fn(all_logits, labels.float())

        return AttackerModelOutput(
            loss = loss,
            logits = atk_logits,
            others = {
                "reasoner_output": res_out,
            }
        )


