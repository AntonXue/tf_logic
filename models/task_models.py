from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from .common import *

""" We wrap some seqcls models to have properly named parameters in their forwrad function
    to conform to what comes out of the datasets.
    This is because HF Trainer does crazy stuff with kwargs
"""


""" One-shot models """

@dataclass
class OneShotTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    seqcls_output: Optional[ModelOutput] = None

class OneShotTaskModel(nn.Module):
    def __init__(self, seqcls_model: SeqClsModel):
        super().__init__()
        self.seqcls_model = seqcls_model

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        seqcls_model_kwargs: dict = {}
    ):
        seqcls_out = self.seqcls_model(tokens.float(), labels=labels, **seqcls_model_kwargs)
        return OneShotTaskOutput(
            loss = seqcls_out.loss,
            logits = seqcls_out.logits,
            seqcls_output = seqcls_out
        )


""" One-shot models with string input """

@dataclass
class OneShotStringTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    seqcls_output: Optional[ModelOutput] = None


class OneShotStringTaskModel(nn.Module):
    def __init__(self, seqcls_model: SeqClsModel):
        super().__init__()
        self.seqcls_model = seqcls_model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        seqcls_model_kwargs: dict = {}
    ):
        seqcls_out = self.seqcls_model(
            input_ids,
            labels = labels,
            use_input_ids = True,
            attention_mask = attention_mask,
            **seqcls_model_kwargs)

        return NextStateTaskOutput(
            loss = seqcls_out.loss,
            logits = seqcls_out.logits,
            seqcls_output = seqcls_out
        )


""" Next-state models """

@dataclass
class NextStateTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    seqcls_output: Optional[ModelOutput] = None


class NextStateTaskModel(nn.Module):
    def __init__(self, seqcls_model: SeqClsModel):
        super().__init__()
        self.seqcls_model = seqcls_model

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        seqcls_model_kwargs: dict = {}
    ):
        seqcls_out = self.seqcls_model(tokens.float(), labels=labels, **seqcls_model_kwargs)
        return NextStateTaskOutput(
            loss = seqcls_out.loss,
            logits = seqcls_out.logits,
            seqcls_output = seqcls_out
        )


""" Autoregressive models """

@dataclass
class AutoregKStepsTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_seqcls_outputs: Optional[tuple[ModelOutput]] = None


class AutoregKStepsTaskModel(nn.Module):
    def __init__(self, seqcls_model: SeqClsModel, num_steps: int):
        super().__init__()
        self.seqcls_model = seqcls_model
        self.num_steps = num_steps
        self.state_to_token = nn.Linear(seqcls_model.num_labels, seqcls_model.input_dim)

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        seqcls_model_kwargs: dict = {}
    ):

        all_succ_logits = ()
        all_seqcls_outs = ()
        all_tokens = tokens

        for t in range(self.num_steps):
            seqcls_out = self.seqcls_model(all_tokens.float(), labels=None, **seqcls_model_kwargs)
            all_seqcls_outs = all_seqcls_outs + (seqcls_out,)

            succ_logits = seqcls_out.logits.view(-1,1,self.seqcls_model.num_labels)  # (N,1,n)
            all_succ_logits = all_succ_logits + (succ_logits,)

            succ_state = (succ_logits > 0)
            succ_state_token = self.state_to_token(succ_state.float())
            all_tokens = torch.cat([all_tokens, succ_state_token], dim=1).long()

        all_succ_logits = torch.cat(all_succ_logits, dim=1)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(all_succ_logits.sigmoid(), labels.float()).to(tokens.device)

        return AutoregKStepsTaskOutput(
            loss = loss,
            logits = all_succ_logits,
            all_seqcls_outputs = all_seqcls_outs
        )


