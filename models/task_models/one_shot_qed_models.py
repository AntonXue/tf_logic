from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from ..common import *

@dataclass
class OneShotQedTaskConfig(TaskConfig):
    num_vars: int
    seqcls_model: nn.Module
    max_seq_len: int = 1024

    def __post_init__(self):
        assert hasattr(self.seqcls_model, "embed_dim")
        assert hasattr(self.seqcls_model, "num_labels")
        assert self.seqcls_model.num_labels == 2


@dataclass
class OneShotQedTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    seqcls_output: Optional[ModelOutput] = None


class OneShotQedEmbedsTaskModel(TaskModel):
    """ One-shot QED task (!!! ambitious !!!) """
    def __init__(self, config: OneShotQedTaskConfig):
        super().__init__(config)
        self.num_tags = num_tags = 4
        self.cls_tag = cls_tag = e(0, num_tags)
        self.sep_tag = sep_tag = e(1, num_tags)
        self.rule_tag = e(2, num_tags)
        self.thm_tag = e(3, num_tags)

        n2_zeros = torch.zeros(2 * config.num_vars)
        self.cls_token = torch.cat([n2_zeros, cls_tag])
        self.sep_token = torch.cat([n2_zeros, sep_tag])

        self.token_dim = 2 * config.num_vars + num_tags
        self.embed_dim = self.seqcls_model.embed_dim
        self.encoder = nn.Linear(self.token_dim, self.embed_dim)
        self.positional_embedding = nn.Embedding(config.max_seq_len, self.embed_dim)

    def _prepare_input_tokens(self, rules: torch.Tensor, theorem: torch.Tensor):
        N, r, _ = rules.shape
        rules_tags = self.rule_tag.view(1,1,-1).repeat(N,r,1).to(rules.device)
        rules_tokens = torch.cat([rules, rules_tags], dim=2)    # (N,r,token_dim)
        thm_tag = self.thm_tag.view(1,-1).repeat(N,1).to(theorem.device)
        thm_token = torch.cat([torch.zeros_like(theorem), theorem, thm_tag], dim=1)
        thm_token = thm_token.view(N,1,-1)  # (N,r,token_dim)
        cls_token = self.cls_token.view(1,1,-1).repeat(N,1,1).to(rules.device) # (N,1,token_dim)
        sep_token = self.sep_token.view(1,1,-1).repeat(N,1,1).to(rules.device) # (N,1,token_dim)
        return torch.cat([cls_token, rules_tokens, sep_token, thm_token], dim=1) # (N,r+3,token_dim)

    def forward(
        self,
        rules: torch.LongTensor,
        theorem: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        seqcls_model_kwargs: Optional[dict] = None
    ):
        """ rules: (N, r, 2n), theorem: (N,n), labels: (N,) """
        device = rules.device
        seqcls_model_kwargs = default(seqcls_model_kwargs, {})

        tokens = self._prepare_input_tokens(rules, theorem)
        x = self.encoder(tokens) # (batch_size, seq_len, embed_dim)
        pos_embeds = self.positional_embedding(torch.arange(0, x.size(1)).to(device))
        x = x + pos_embeds.view(1, x.size(1), -1)

        seqcls_out = self.seqcls_model(x, labels=labels, **seqcls_model_kwargs)
        return OneShotQedTaskOutput(
            loss = seqcls_out.loss,
            logits = seqcls_out.logits,
            seqcls_output = seqcls_out)


class OneShotQedStringTaskModel(TaskModel):
    pass

