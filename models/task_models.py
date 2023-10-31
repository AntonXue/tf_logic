from typing import Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import ModelOutput

from .utils import *

""" One-shot QED """

@dataclass
class QedTaskConfig(TaskConfig):
    num_vars: int
    num_rules: int
    seq2seq_model: MySeq2SeqModel
    max_seq_len: int = 1024

    def __post_init__(self):
        if self.seq2seq_model.max_seq_len:
            assert self.max_seq_len <= self.seq2seq_model.max_seq_len


class QedTaskModel(BaseTaskModel):
    """ One-shot QED task (!!! ambitious !!!)
    """
    def __init__(self, seq2seq_model: MySeq2SeqModel, config: QedTaskConfig):
        # super call does __setattr__for configs
        super().__init__(config)

        self.num_tags = num_tags = 4
        self.cls_tag = cls_tag = e(0, num_tags)
        self.sep_tag = sep_tag = e(1, num_tags)
        self.rule_tag = e(2, num_tags)
        self.thm_tag = e(3, num_tags)

        n2_zeros = torch.zeros(2 * self.num_vars)
        self.cls_token = torch.cat([n2_zeros, cls_tag])
        self.sep_token = torch.cat([n2_zeros, sep_tag])

        self.token_dim = token_dim = 2 * self.num_vars + num_tags
        self.embed_dim = embed_dim = seq2seq_model.embed_dim
        self.encoder = nn.Sequential(
                nn.Linear(token_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim))

        self.qed_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1))

        self.positional_embedding = nn.Embedding(self.max_seq_len, embed_dim)

    def _tokenize_rules(self, rules: torch.Tensor):
        N, r, _ = rules.shape
        tags = self.rule_tag.view(1,1,-1).repeat(N,r,1).to(rules.device)
        rule_tokens = torch.cat([rules, tags], dim=2)
        return rule_tokens  # (N,r,token_dim)

    def _tokenize_theorem(self, theorem: torch.Tensor):
        N, n = theorem.shape
        tags = self.thm_tag.view(1,-1).repeat(N,1).to(theorem.device)
        thm_token = torch.cat([torch.zeros_like(theorem), theorem, tags], dim=1)
        return thm_token.view(N,1,-1)   # (N,1,token_dim)

    def _prepare_input_tokens(self, rules: torch.Tensor, theorem: torch.Tensor):
        N, _, _ = rules.shape
        cls_token = self.cls_token.view(1,1,-1).repeat(N,1,1).to(rules.device) # (N,1,d)
        sep_token = self.sep_token.view(1,1,-1).repeat(N,1,1).to(rules.device) # (N,1,d)
        rule_tokens = self._tokenize_rules(rules)       # (N,r,d)
        thm_token = self._tokenize_theorem(theorem)     # (N,1,d)     
        x = torch.cat([cls_token, rule_tokens, sep_token, thm_token], dim=1)
        return x    # (N,r+3,d)

    def forward(
            self,
            rules: torch.Tensor,
            theorem: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            seq2seq_model_kwargs: Optional[dict] = None,
            include_seq2seq_output: Optional[bool] = None):

        tokens = self._prepare_input_tokens(rules, theorem)
        x = self.encoder(tokens) # (batch_size, seq_len, embed_dim)
        pos_embeds = self.positional_embedding(torch.arange(0, x.size(1)))
        x = x + pos_embeds.view(1, x.size(1), -1)

        if seq2seq_model_kwargs:
            seq2seq_out = self.seq2seq_model(x, **seq2seq_model_kwargs)
        else:
            seq2seq_out = self.seq2seq_model(x)

        qeds = self.qed_head(seq2seq_out["tensor"])
        logits = qeds[:,0].view(-1)

        loss = None
        if targets is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(logits.sigmoid(), targets.float().to(logits.device))

        return ModelOutput(loss = loss,
                           logits = logits,
                           seq2seq_tensor = seq2seq_out["tensor"],
                           seq2seq_out = seq2seq_out if include_seq2seq_output else None)


""" One-step tasks """

@dataclass
class OneStepTaskConfig(TaskConfig):
    pass

class StepTaskModel(BaseTaskModel):
    """ Check whether we can get s' = one_step(rules, s)
    """
    def __init__(self, config: OneStepTaskConfig):
        super().__init__(config)

    def forward(self, rules, theorem, targets=None):
        pass

""" Autoregressive stuff """

@dataclass
class AutoRegStepTaskConfig(TaskConfig):
    num_steps: Optional[int] = None


class AutoRegStepTaskModel(BaseTaskModel):
    """ Autoregressive task stuff
    """
    def __init__(self, config: AutoRegStepTaskConfig):
        super().__init__(config)

    def forward(self, rules, theorem, targets=None):
        pass


