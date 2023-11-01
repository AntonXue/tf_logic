from typing import Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import ModelOutput
from transformers.modeling_outputs import *

from .utils import *


""" Some definition for tasks """

@dataclass
class BaseTaskConfig: pass

class BaseTaskModel(nn.Module):
    def __init__(self, config: BaseTaskConfig):
        super().__init__()
        self.config = config
        # Also copy over all the rules
        for k, v in asdict(config).items():
            self.__setattr__(k, v)

""" One-shot QED """

@dataclass
class OneShotQedTaskConfig(BaseTaskConfig):
    num_rules: int
    num_vars: int
    seq2seq_model: MySeq2SeqModel
    max_seq_len: int = 1024

    def __post_init__(self):
        if self.seq2seq_model.max_seq_len:
            assert self.max_seq_len <= self.seq2seq_model.max_seq_len

@dataclass
class OneShotQedTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    seq2seq_output: Optional[ModelOutput] = None


class OneShotQedTaskModel(BaseTaskModel):
    """ One-shot QED task (!!! ambitious !!!) """
    def __init__(self, config: OneShotQedTaskConfig):
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
        self.embed_dim = embed_dim = self.seq2seq_model.embed_dim
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
            seq2seq_model_kwargs: Optional[dict] = None,
            include_seq2seq_output: Optional[bool] = None):
        """ rules: (N,r,2n)
            theorem: (N,n)
            labels: (N,)
        """
        tokens = self._prepare_input_tokens(rules, theorem)
        x = self.encoder(tokens) # (batch_size, seq_len, embed_dim)
        pos_embeds = self.positional_embedding(torch.arange(0, x.size(1)).to(x.device))
        x = x + pos_embeds.view(1, x.size(1), -1)

        if seq2seq_model_kwargs:
            seq2seq_out = self.seq2seq_model(x, **seq2seq_model_kwargs)
        else:
            seq2seq_out = self.seq2seq_model(x)

        qeds = self.qed_head(seq2seq_out["last_hidden_state"])
        logits = qeds[:,0].view(-1)

        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            preds = (logits > 0).long()
            loss = loss_fn(logits.sigmoid(), labels.float().to(logits.device))

        # return SequenceClassifierOutput(
        return OneShotQedTaskOutput(
                loss = loss,
                logits = logits,
                seq2seq_output = seq2seq_out)


""" One-step tasks """

@dataclass
class PredictSuccTaskConfig(BaseTaskConfig):
    num_rules: int
    num_vars: int
    seq2seq_model: MySeq2SeqModel
    max_seq_len: int = 1024

    def __post_init__(self):
        if self.seq2seq_model.max_seq_len:
            assert self.max_seq_len <= self.seq2seq_model.max_seq_len


@dataclass
class PredictSuccTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    succ: Optional[torch.FloatTensor] = None
    seq2seq_output: Optional[ModelOutput] = None

class PredictSuccTaskModel(BaseTaskModel):
    """ Check whether we can get s' = one_step(rules, s) """
    def __init__(self, config: PredictSuccTaskConfig):
        super().__init__(config)
        self.num_tags = num_tags = 4
        self.cls_tag = cls_tag = e(0, num_tags)
        self.sep_tag = sep_tag = e(1, num_tags)
        self.rule_tag = e(2, num_tags)
        self.state_tag = e(3, num_tags)

        n2_zeros = torch.zeros(2 * self.num_vars)
        self.cls_token = torch.cat([n2_zeros, cls_tag])
        self.sep_token = torch.cat([n2_zeros, sep_tag])

        self.token_dim = token_dim = 2 * self.num_vars + num_tags
        self.embed_dim = embed_dim = self.seq2seq_model.embed_dim
        self.encoder = nn.Sequential(
                nn.Linear(token_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim))

        self.succ_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.num_vars))

        self.positional_embedding = nn.Embedding(self.max_seq_len, embed_dim)

    def _prepare_input_tokens(self, rules: torch.Tensor, state: torch.Tensor):
        N, r, _ = rules.shape

        rules_tags = self.rule_tag.view(1,1,-1).repeat(N,r,1).to(rules.device)
        rules_tokens = torch.cat([rules, rules_tags], dim=2)    # (N,r,token_dim)

        state_tag = self.state_tag.view(1,-1).repeat(N,1).to(state.device)
        state_token = torch.cat([torch.zeros_like(state), state, state_tag], dim=1)
        state_token = state_token.view(N,1,-1)  # (N,r,token_dim)

        cls_token = self.cls_token.view(1,1,-1).repeat(N,1,1).to(rules.device) # (N,1,token_dim)
        sep_token = self.sep_token.view(1,1,-1).repeat(N,1,1).to(rules.device) # (N,1,token_dim)
        return torch.cat([cls_token, rules_tokens, sep_token, state_token], dim=1) # (N,r+3,token_dim)

    def forward(
            self,
            rules: torch.LongTensor,
            state: torch.LongTensor,
            labels: Optional[torch.LongTensor] = None,
            seq2seq_model_kwargs: Optional[dict] = None,
            include_seq2seq_output: Optional[bool] = None):
        """ rules: (N,r,2n)
            state: (N,n)
            labels: (N,n)
        """
        tokens = self._prepare_input_tokens(rules, state)
        x = self.encoder(tokens) # (batch_size, seq_len, embed_dim)
        pos_embeds = self.positional_embedding(torch.arange(0, x.size(1)).to(x.device))
        x = x + pos_embeds.view(1, x.size(1), -1)

        if seq2seq_model_kwargs:
            seq2seq_out = self.seq2seq_model(x, **seq2seq_model_kwargs)
        else:
            seq2seq_out = self.seq2seq_model(x)

        succ = self.succ_head(seq2seq_out["last_hidden_state"])  # (batch_size, seq_len, n)
        succ = succ[:,0]    # (batch_size, n)
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(succ.sigmoid(), labels.float().to(succ.device))

        return PredictSuccTaskOutput(
                loss = loss,
                succ = succ,
                seq2seq_output = seq2seq_out)



""" Autoregressive stuff """

@dataclass
class AutoRegStepTaskConfig(BaseTaskConfig):
    num_steps: Optional[int] = None


class AutoRegStepTaskModel(BaseTaskModel):
    """ Autoregressive task stuff
    """
    def __init__(self, config: AutoRegStepTaskConfig):
        super().__init__(config)

    def forward(self, rules, theorem, labels=None):
        pass


