from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from ..common import *

@dataclass
class OneShotTFLConfig(TFLConfig):
    num_vars: int
    max_seq_len: int = 1024


@dataclass
class OneShotTFLOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    seqcls_output: Optional[ModelOutput] = None


class OneShotEmbedsTFLModel(TFLModel):
    """ One-shot QED task (!!! ambitious !!!) """
    def __init__(self, seqcls_model: nn.Module, config: OneShotTFLConfig):
        super().__init__(seqcls_model, config)
        self.num_tags = 4
        self.token_dim = 2 * self.num_vars + self.num_tags
        self.cls_tag = e(0, self.token_dim)
        self.sep_tag = e(1, self.token_dim)
        self.rule_tag = e(2, self.token_dim)
        self.theorem_tag = e(3, self.token_dim)

        self.encoder = nn.Linear(self.token_dim, self.embed_dim)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)

    def _prepare_input_tokens(self, rules: torch.Tensor, theorem: torch.Tensor):
        N, r, _ = rules.shape
        device = rules.device
        cls_token = self.cls_tag.view(1,1,-1).repeat(N,1,1).to(device)
        rules_tokens = F.pad(rules, (self.num_tags, 0)) + self.rule_tag.view(1,1,-1).to(device)
        thm_token = F.pad(theorem.view(N,1,-1), (self.num_tags + self.num_vars, 0)) + \
                self.theorem_tag.view(1,1,-1).to(device)
        cls_token = self.cls_tag.view(1,1,-1).repeat(N,1,1).to(device)
        sep_token = self.sep_tag.view(1,1,-1).repeat(N,1,1).to(device)
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
        x = self.encoder(tokens.float()) # (batch_size, seq_len, embed_dim)
        pos_embeds = self.pos_embedding(torch.arange(0, x.size(1)).to(device))
        x = x + pos_embeds.view(1, x.size(1), -1)

        seqcls_out = self.seqcls_model(x, labels=labels, **seqcls_model_kwargs)
        return OneShotTFLOutput(
            loss = seqcls_out.loss,
            logits = seqcls_out.logits,
            seqcls_output = seqcls_out)


class OneShotStringTFLModel(TFLModel):
    pass

