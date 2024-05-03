from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from .common import *
from .seqcls_models.my_models import *

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


class OneShotTaskModel(SeqClsModel):
    def __init__(self, seqcls_model: SeqClsModel):
        super().__init__()
        self.seqcls_model = seqcls_model

    @property
    def model_name(self):
        return self.seqcls_model.model_name

    @property
    def input_dim(self):
        return self.seqcls_model.input_dim

    @property
    def embed_dim(self):
        return self.seqcls_model.embed_dim

    @property
    def num_labels(self):
        return self.seqcls_model.num_labels

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        seqcls_out = self.seqcls_model(tokens.float(), labels=labels, **kwargs)
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


class OneShotStringTaskModel(SeqClsModel):
    def __init__(self, seqcls_model: SeqClsModel):
        super().__init__()
        self.seqcls_model = seqcls_model

    @property
    def model_name(self):
        return self.seqcls_model.model_name

    @property
    def input_dim(self):
        return self.seqcls_model.input_dim

    @property
    def embed_dim(self):
        return self.seqcls_model.embed_dim

    @property
    def num_labels(self):
        return self.seqcls_model.num_labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        seqcls_out = self.seqcls_model(
            input_ids = input_ids,
            labels = labels,
            attention_mask = attention_mask,
            **kwargs)

        return OneShotStringTaskOutput(
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


class AutoregKStepsTaskModel(SeqClsModel):
    """ train_supervision_mode
        * all: supervise on all the outputs
        * final: only supervise the final output
    """
    def __init__(
        self,
        seqcls_model: SeqClsModel,
        num_steps: int,
        train_supervision_mode: str = "all"
    ):
        super().__init__()
        self.seqcls_model = seqcls_model
        self.num_steps = num_steps
        self.state_to_token = nn.Linear(seqcls_model.num_labels, seqcls_model.input_dim)
        assert train_supervision_mode in ["all", "final"]
        self.train_supervision_mode = train_supervision_mode
        self.loss_fn = nn.BCEWithLogitsLoss()

    @property
    def model_name(self):
        return self.seqcls_model.model_name

    @property
    def input_dim(self):
        return self.seqcls_model.input_dim

    @property
    def embed_dim(self):
        return self.seqcls_model.embed_dim

    @property
    def num_labels(self):
        return self.seqcls_model.num_labels

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
            tokens: (batch_size, seq_len, token_dim)
        """
        all_succ_logits = ()
        all_seqcls_outs = ()
        all_tokens = tokens

        for t in range(self.num_steps):
            seqcls_out = self.seqcls_model(
                all_tokens.float(),
                labels = None,
                **kwargs
            )
            all_seqcls_outs += (seqcls_out,)

            succ_logits = seqcls_out.logits # (N,n)
            all_succ_logits += (succ_logits,)

            succ = (succ_logits > 0)  # (N,n)
            succ_token = self.state_to_token(succ.unsqueeze(1).float()) # (N,1,token_dim)

            # Prepare for next round of generation
            all_tokens = torch.cat([all_tokens, succ_token], dim=1).float()

        all_succ_logits = torch.stack(all_succ_logits, dim=1) # (N,num_steps,n)

        loss = None
        if labels is not None:
            if self.train_supervision_mode == "all":
                loss = self.loss_fn(all_succ_logits, labels.float()).to(tokens.device)
            elif self.train_supervision_mode == "final":
                loss = self.loss_fn(all_succ_logits[:,-1], labels[:,-1].float()).to(tokens.device)
            else:
                raise ValueError(f"Unknown train_supervision_mode: {self.train_supervision_mode}")

        return AutoregKStepsTaskOutput(
            loss = loss,
            logits = all_succ_logits,
            all_seqcls_outputs = all_seqcls_outs
        )


