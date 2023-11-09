from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from ..common import *

@dataclass
class AutoRegFixedStepsTaskConfig(TaskConfig):
    num_vars: int
    num_steps: int
    seqcls_model: nn.Module
    max_seq_len: int = 1024

    def __post_init__(self):
        assert hasattr(self.seqcls_model, "embed_dim")
        assert hasattr(self.seqcls_model, "num_labels")
        assert self.num_vars == self.seqcls_model.num_labels


@dataclass
class AutoRegFixedStepsTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    all_seqcls_outputs: Optional[Tuple[ModelOutput]] = None


class AutoRegFixedStepsEmbedsTaskModel(TaskModel):
    """ Autoregressive task stuff
    """
    def __init__(self, config: AutoRegFixedStepsTaskConfig):
        super().__init__(config)
        self.num_tags = num_tags = 4
        self.cls_tag = cls_tag = e(0, num_tags)
        self.sep_tag = sep_tag = e(1, num_tags)
        self.rule_tag = e(2, num_tags)
        self.state_tag = e(3, num_tags)

        n2_zeros = torch.zeros(2 * config.num_vars)
        self.cls_token = torch.cat([n2_zeros, cls_tag])
        self.sep_token = torch.cat([n2_zeros, sep_tag])

        self.token_dim = 2 * config.num_vars + num_tags
        self.embed_dim = self.seqcls_model.embed_dim
        self.encoder = nn.Linear(self.token_dim, self.embed_dim)

        self.positional_embedding = nn.Embedding(config.max_seq_len, self.embed_dim)

    def _tokenize_rules(self, rules: torch.Tensor):
        N, r, _ = rules.shape
        tags = self.rule_tag.view(1,1,-1).repeat(N,r,1).to(rules.device)
        rules_tokens = torch.cat([rules, tags], dim=2) # (N,r,token_dim)
        return rules_tokens

    def _tokenize_states(self, states: torch.Tensor):
        N, k, n = states.shape
        tags = self.state_tag.view(1,1,-1).repeat(N,k,1).to(states.device)
        states_tokens = torch.cat([torch.zeros_like(states), states, tags], dim=2) # (N,k,token_dim)
        return states_tokens

    def forward(
            self,
            rules: torch.LongTensor,
            state: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            seqcls_model_kwargs: Optional[dict] = None):
        """ rules: (N,r,2n), state: (N,n) """
        N, _, _ = rules.shape
        device = rules.device
        state = default(state, torch.zeros(N,self.config.num_vars).long().to(device))
        seqcls_model_kwargs = default(seqcls_model_kwargs, {})

        cls_token = self.cls_token.view(1,1,-1).repeat(N,1,1).to(device)    # (N,1,token_dim)
        sep_token = self.sep_token.view(1,1,-1).repeat(N,1,1).to(device)    # (N,1,token_dim)
        rules_tokens = self._tokenize_rules(rules)  # (N,r,token_dim)
        state_token = self._tokenize_states(state.view(N,1,-1)) # (N,1,token_dim)

        # We will append to these token sequences as we run things
        all_succ_logits = ()    # This will grow
        all_seq_tokens = (cls_token, rules_tokens, sep_token, state_token)
        all_seqcls_outs = ()

        for t in range(self.num_steps):
            x = self.encoder(torch.cat(all_seq_tokens, dim=1))
            pos_embeds = self.positional_embedding(torch.arange(0, x.size(1)).to(device))
            x = x + pos_embeds.view(1, x.size(1), -1)

            # We do NOT pass labels here quite yet
            seqcls_out = self.seqcls_model(x, labels=None, **seqcls_model_kwargs)
            all_seqcls_outs = all_seqcls_outs + (seqcls_out,)

            succ_logits = seqcls_out.logits.view(N,1,-1)    # (N,1,n)
            all_succ_logits = all_succ_logits + (succ_logits,)

            succ_state = (succ_logits > 0).long()   # (N,1,n)
            succ_state_token = self._tokenize_states(succ_state)
            all_seq_tokens = all_seq_tokens + (succ_state_token,)

        all_succ_logits = torch.cat(all_succ_logits, dim=1)
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(all_succ_logits.sigmoid(), labels.float().to(device))

        return AutoRegFixedStepsTaskOutput(
                loss = loss,
                logits = all_succ_logits,
                all_seqcls_outputs = all_seqcls_outs)


class AutoRegFixedStepsStringTaskModel(TaskModel):
    pass


