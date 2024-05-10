from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
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


@dataclass
class TheoryAutoregKStepsTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    attentions: Tuple[torch.FloatTensor] = None


class TheoryAutoregKStepsModel(nn.Module):
    """ The theoretical model's algorithm """
    def __init__(
        self,
        num_vars: int,
        num_steps: int,
        lambd: Optional[float] = None,
        rho: Optional[float] = None,
        logit_type: str = "continuous"
    ):
        super().__init__()
        n = num_vars
        self.num_vars = n
        self.embed_dim = 2 * n
        self.num_steps = num_steps
        self.lambd = n**2 if lambd is None else lambd
        self.rho = n**2 if rho is None else rho

        self.Pia = torch.cat([torch.eye(n), torch.zeros(n,n)], dim=1)
        self.Pib = torch.cat([torch.zeros(n,n), torch.eye(n)], dim=1)

        self.Wq = nn.Linear(2*n, n, bias=True)
        self.Wq.weight.data = self.Pib
        self.Wq.bias.data = -1*torch.ones(n)
        for p in self.Wq.parameters():
            p.requires_grad = False

        self.Wk = nn.Linear(2*n, n, bias=False)
        self.Wk.weight.data = self.lambd * self.Pia
        for p in self.Wk.parameters():
            p.requires_grad = False

        self.Wv = nn.Linear(2*n, 2*n, bias=False)
        self.Wv.weight.data = self.rho * torch.matmul(self.Pib.transpose(0,1), self.Pib)
        for p in self.Wv.parameters():
            p.requires_grad = False

        # The continuous piecewise linear binarization function
        self.logit_type = logit_type
        if logit_type == "binarized":
            self.id_ffwd = lambda x: 3*F.relu(x - 1/3) - 3*F.relu(x - 2/3)
        elif logit_type == "continuous":
            # self.id_ffwd = lambda z: z - 1/2
            self.id_ffwd = lambda z: F.tanh(z - 1/2)
            # self.id_ffwd = lambda z: torch.log(n*F.relu(z) + 1e-4)
            # self.id_ffwd = lambda z: torch.log(2 * F.relu(z - 1/2) + 1e-5)
            # self.id_ffwd = lambda z: torch.log(F.relu(z-0.5) + 1e-4) - torch.log(F.relu(1-z+0.5) + 1e-4)
        else:
            raise ValueError(f"Unrecognized logit_type {logit_type}")
        self.loss_fn = nn.BCEWithLogitsLoss()

    @property
    def model_name(self):
        return f"theory"

    @property
    def input_dim(self):
        return self.embed_dim # This is also the input dim

    @property
    def num_labels(self):
        return self.num_vars

    def one_step(self, x: torch.FloatTensor, output_attentions: bool = False):
        N, L, n2 = x.shape
        x = x.float()
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        A = F.softmax(torch.bmm(Q, K.transpose(-1,-2)), dim=-1) # (N,L,L)

        # Output of attention thing
        z = x + torch.bmm(A, V)
        y = self.id_ffwd(z)
        logits = y[:,-1,self.num_vars:]

        return (logits, A) if output_attentions else logits


    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ):
        all_succ_logits = ()
        all_attentions = () if output_attentions else None
        all_tokens = tokens

        for t in range(self.num_steps):
            if output_attentions:
                logits, attns = self.one_step(all_tokens.float(), output_attentions=True)
                all_attentions += (attns,)
            else:
                logits = self.one_step(all_tokens.float(), output_attentions=False)

            all_succ_logits += (logits,)
            succ = (logits > 0).long()
            succ_token = torch.cat([torch.zeros_like(succ), succ], dim=-1)
            all_tokens = torch.cat([all_tokens, succ_token.unsqueeze(1)], dim=1).float()

        all_succ_logits = torch.stack(all_succ_logits, dim=1) # (N, num_steps, n)

        loss = None
        if labels is not None:
            loss = self.loss_fn(all_succ_logits.float(), labels.float())

        return TheoryAutoregKStepsTaskOutput(
            loss = loss,
            logits = all_succ_logits,
            attentions = all_attentions,
        )


