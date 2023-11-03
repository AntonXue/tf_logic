from typing import Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import ModelOutput
from transformers.modeling_outputs import *

from .common import *


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
            seq2seq_model_kwargs: Optional[dict] = None):
        """ rules: (N, r, 2n), theorem: (N,n), labels: (N,) """
        device = rules.device
        seq2seq_model_kwargs = default(seq2seq_model_kwargs, {})

        tokens = self._prepare_input_tokens(rules, theorem)
        x = self.encoder(tokens) # (batch_size, seq_len, embed_dim)
        pos_embeds = self.positional_embedding(torch.arange(0, x.size(1)).to(device))
        x = x + pos_embeds.view(1, x.size(1), -1)

        seq2seq_out = self.seq2seq_model(x, **seq2seq_model_kwargs)

        qeds = self.qed_head(seq2seq_out["last_hidden_state"])
        logits = qeds[:,0].view(-1)
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            preds = (logits > 0).long()
            loss = loss_fn(logits.sigmoid(), labels.float().to(device))

        # return SequenceClassifierOutput(
        return OneShotQedTaskOutput(
                loss = loss,
                logits = logits,
                seq2seq_output = seq2seq_out)


""" One-step tasks """

@dataclass
class PredictSuccTaskConfig(BaseTaskConfig):
    num_vars: int
    seq2seq_model: MySeq2SeqModel
    max_seq_len: int = 1024

    def __post_init__(self):
        if self.seq2seq_model.max_seq_len:
            assert self.max_seq_len <= self.seq2seq_model.max_seq_len


@dataclass
class PredictSuccTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
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
            seq2seq_model_kwargs: Optional[dict] = None):
        """ rules: (N,r,2n), state: (N,n), labels: (N,n) """
        device = rules.device
        seq2seq_model_kwargs = default(seq2seq_model_kwargs, {})

        tokens = self._prepare_input_tokens(rules, state)
        x = self.encoder(tokens) # (batch_size, seq_len, embed_dim)
        pos_embeds = self.positional_embedding(torch.arange(0, x.size(1)).to(device))
        x = x + pos_embeds.view(1, x.size(1), -1)

        seq2seq_out = self.seq2seq_model(x, **seq2seq_model_kwargs)

        succ = self.succ_head(seq2seq_out["last_hidden_state"])  # (batch_size, seq_len, n)
        succ = succ[:,0]    # (batch_size, n)
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(succ.sigmoid(), labels.float().to(device))

        return PredictSuccTaskOutput(
                loss = loss,
                logits = succ,
                seq2seq_output = seq2seq_out)



""" Autoregressive stuff """

@dataclass
class AutoRegFixedStepsTaskConfig(BaseTaskConfig):
    num_vars: int
    num_steps: int
    seq2seq_model: MySeq2SeqModel
    max_seq_len: int = 1024

    def __post_init__(self):
        if self.seq2seq_model.max_seq_len:
            assert self.max_seq_len <= self.seq2seq_model.max_seq_len

@dataclass
class AutoRegFixedStepsTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    all_seq2seq_outputs: Optional[Tuple[ModelOutput]] = None


class AutoRegFixedStepsTaskModel(BaseTaskModel):
    """ Autoregressive task stuff
    """
    def __init__(self, config: AutoRegFixedStepsTaskConfig):
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
            seq2seq_model_kwargs: Optional[dict] = None):
        """ rules: (N,r,2n), state: (N,n) """
        N, _, _ = rules.shape
        device = rules.device
        state = default(state, torch.zeros(N,self.num_vars).long().to(device))
        seq2seq_model_kwargs = default(seq2seq_model_kwargs, {})

        cls_token = self.cls_token.view(1,1,-1).repeat(N,1,1).to(device)    # (N,1,token_dim)
        sep_token = self.sep_token.view(1,1,-1).repeat(N,1,1).to(device)    # (N,1,token_dim)
        rules_tokens = self._tokenize_rules(rules)  # (N,r,token_dim)
        state_token = self._tokenize_states(state.view(N,1,-1)) # (N,1,token_dim)

        # We will append to these token sequences as we run things
        all_succ_logits = ()    # This will grow
        all_seq_tokens = (cls_token, rules_tokens, sep_token, state_token)
        all_seq2seq_outs = ()

        for t in range(self.num_steps):
            x = self.encoder(torch.cat(all_seq_tokens, dim=1))
            pos_embeds = self.positional_embedding(torch.arange(0, x.size(1)).to(device))
            x = x + pos_embeds.view(1, x.size(1), -1)

            seq2seq_out = self.seq2seq_model(x, **seq2seq_model_kwargs)
            all_seq2seq_outs = all_seq2seq_outs + (seq2seq_out,)

            succ_logits = self.succ_head(seq2seq_out["last_hidden_state"])
            succ_logits = succ_logits[:,0].view(N,1,-1) # (N,1,n)
            all_succ_logits = all_succ_logits + (succ_logits,)

            succ_state = (succ_logits > 0).long()   # (N,1,n)
            succ_state_token = self._tokenize_states(succ_state)
            all_seq_tokens = all_seq_tokens + (succ_state_token,)

        all_succ_logits = torch.cat(all_succ_logits, dim=1)
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(all_succ_logits.sigmoid(), labels.float().to(device))

        return AutoRegFixedStepsTaskOutput(
                loss = loss,
                logits = all_succ_logits,
                all_seq2seq_outputs = all_seq2seq_outs)


