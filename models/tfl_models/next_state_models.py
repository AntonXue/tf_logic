from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from ..common import *

@dataclass
class NextStateTFLConfig(TFLConfig):
    num_vars: int
    max_seq_len: int = 1024


@dataclass
class NextStateTFLOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    seqcls_output: Optional[ModelOutput] = None


class NextStateEmbedsTFLModel(TFLModel):
    """ Check whether we can get s' = one_step(rules, s) """
    def __init__(self, seqcls_model: nn.Module, config: NextStateTFLConfig):
        super().__init__(seqcls_model, config)
        self.num_tags = 4
        self.token_dim = 2 * self.num_vars + self.num_tags
        self.cls_tag = e(0, self.token_dim)
        self.sep_tag = e(1, self.token_dim)
        self.rule_tag = e(2, self.token_dim)
        self.state_tag = e(3, self.token_dim)
        self.fpad_shape = (self.num_tags, 0)
        self.encoder = nn.Linear(self.token_dim, self.embed_dim)
        self.positional_embedding = nn.Embedding(config.max_seq_len, self.embed_dim)

    def _prepare_input_tokens(self, rules: torch.Tensor, state: torch.Tensor):
        N, r, _ = rules.shape
        device = rules.device
        rules_token = F.pad(rules, self.fpad_shape) + self.rule_tag.view(1,1,-1).to(device)
        state_token = F.pad(state.view(N,1,-1), (self.num_vars + self.num_tags, 0)) + \
                self.state_tag.view(1,1,-1).to(device)
        cls_token = self.cls_tag.view(1,1,-1).repeat(N,1,1).to(device)
        sep_token = self.sep_tag.view(1,1,-1).repeat(N,1,1).to(device)
        return torch.cat([cls_token, rules_token, sep_token, state_token], dim=1) # (N,r+3,token_dim)

    def forward(
        self,
        rules: torch.LongTensor,
        state: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        seqcls_model_kwargs: Optional[dict] = None
    ):
        """ rules: (N,r,2n), state: (N,n), labels: (N,n) """
        device = rules.device
        seqcls_model_kwargs = default(seqcls_model_kwargs, {})

        tokens = self._prepare_input_tokens(rules, state)
        x = self.encoder(tokens.float()) # (batch_size, seq_len, embed_dim)
        pos_embeds = self.positional_embedding(torch.arange(0, x.size(1)).to(device))
        x = x + pos_embeds.view(1, x.size(1), -1)

        seqcls_out = self.seqcls_model(x, labels=labels, **seqcls_model_kwargs)

        return NextStateTFLOutput(
            loss = seqcls_out.loss,
            logits = seqcls_out.logits,
            seqcls_output = seqcls_out)


class NextStateStringTFLModel(TFLModel):
    pass


