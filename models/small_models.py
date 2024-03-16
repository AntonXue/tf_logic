from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from transformers import GPT2Config, GPT2Model

from .common import *
from .seqcls_models.my_models import *



@dataclass
class SmallSuccOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class MarginLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, labels):
        labels = 2 * labels - 1
        loss = F.relu(1 - pred.float() * labels.float()).mean()
        return loss


class SmallGpt2(nn.Module):
    def __init__(self, num_vars: int, embed_dim: int, loss_fn: str = "bce"):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim
        self.embed_fn = nn.Linear(2*num_vars, embed_dim)
        self.gpt2 = GPT2Model(GPT2Config(
            n_embd = embed_dim,
            n_layer = 1,
            n_head = 1,
        ))
        self.cls_head = nn.Linear(embed_dim, num_vars)
        self.loss_fn = loss_fn

    @property
    def desc_str(self):
        return f"gpt2_{self.loss_fn}_n{self.num_vars}_d{self.embed_dim}"

    def forward(self, tokens: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        x = self.embed_fn(tokens.float())
        out = self.gpt2(inputs_embeds=x)
        logits = self.cls_head(out.last_hidden_state)[:,-1]

        loss = None
        if labels is not None:
            if self.loss_fn == "bce":
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            elif self.loss_fn == "margin":
                loss = MarginLoss()(logits, labels)
            else:
                raise ValueError(f"Unrecognized loss_fn {self.loss_fn}")

        return SmallSuccOutput(
            loss = loss,
            logits = logits
        )


class SmallTfA(nn.Module):
    def __init__(self, num_vars: int, embed_dim: int, loss_fn: str = "bce"):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim
        self.embed_fn = nn.Linear(2 * num_vars, embed_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.cls_head = nn.Linear(embed_dim, num_vars)
        self.loss_fn = loss_fn

    @property
    def desc_str(self):
        return f"tfa_{self.loss_fn}_n{self.num_vars}_d{self.embed_dim}"

    def forward(self, tokens: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        N, L, _ = tokens.shape
        device = tokens.device
        x = self.embed_fn(tokens.float())
        
        mask = torch.triu(torch.ones(L,L), diagonal=1).view(1,L,L).repeat(N,1,1).to(device) * -999
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        wts = F.softmax(torch.bmm(Q, K.transpose(1,2)) + mask, dim=2)
        a = torch.bmm(wts, V)
        y = (x + a) + self.ffwd(x + a)
        logits = self.cls_head(y)[:,-1]

        loss = None
        if labels is not None:
            if self.loss_fn == "bce":
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            elif self.loss_fn == "margin":
                loss = MarginLoss()(logits, labels)
            else:
                raise ValueError(f"Unrecognized loss_fn {self.loss_fn}")

        return SmallSuccOutput(
            loss = loss,
            logits = logits,
        )


class SmallTfB(nn.Module):
    def __init__(self, num_vars: int, loss_fn: str = "bce"):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim = 2 * num_vars + 1
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.cls_head = nn.Linear(embed_dim, num_vars)
        self.loss_fn = loss_fn

    @property
    def desc_str(self):
        return f"tfb_{self.loss_fn}_n{self.num_vars}_d{self.embed_dim}"

    def forward(self, tokens: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        N, L, _ = tokens.shape
        device = tokens.device

        x = torch.cat([torch.ones(N,L,1).to(device), tokens], dim=2).float()
        
        mask = torch.triu(torch.ones(L,L), diagonal=1).view(1,L,L).repeat(N,1,1).to(device) * -999
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        wts = F.softmax(torch.bmm(Q, K.transpose(1,2)) + mask, dim=2)
        a = torch.bmm(wts, V)
        y = (x + a) + self.ffwd(x + a)
        logits = self.cls_head(y)[:,-1]

        loss = None
        if labels is not None:
            if self.loss_fn == "bce":
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            elif self.loss_fn == "margin":
                loss = MarginLoss()(logits, labels)
            else:
                raise ValueError(f"Unrecognized loss_fn {self.loss_fn}")

        return SmallSuccOutput(
            loss = loss,
            logits = logits,
        )



class SmallTfC(nn.Module):
    def __init__(
        self,
        num_vars: int,
        attn_fn: str = "relu",
        use_bias: bool = True,
        loss_fn: str = "bce",
        init_ones: bool = True,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim = 2 * num_vars + 1
        self.Wa = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wb = nn.Linear(embed_dim, num_vars, bias=use_bias)
        self.use_bias = use_bias
        self.loss_fn = loss_fn
        
        self.attn_fn_str = attn_fn
        if attn_fn == "relu":
            self.attn_fn = F.relu
        elif attn_fn == "sigmoid":
            self.attn_fn = F.sigmoid
        elif attn_fn == "softplus":
            self.attn_fn = F.softplus
        else:
            raise ValueError(f"Unrecognized attn_fn {attn_fn}")

        self.init_ones = init_ones
        if init_ones:
            self.Wa.weight.data.fill_(1)
            self.Wb.weight.data.fill_(1)

    @property
    def desc_str(self):
        bstr = "B1" if self.use_bias else "B0"
        iostr = "Init1" if self.init_ones else "InitR"
        return f"tfc_{self.attn_fn_str}_{bstr}_{self.loss_fn}_n{self.num_vars}_d{self.embed_dim}_Init{iostr}"

    def forward(self, tokens: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        N, L, _ = tokens.shape
        device = tokens.device

        x = torch.cat([torch.ones(N,L,1).to(device), tokens], dim=2).float()
        wts = self.attn_fn(torch.bmm(self.Wa(x), x.transpose(1,2)))
        a = torch.bmm(wts, x)
        y = self.Wb(a)

        logits = y[:,-1]
        
        loss = None
        if labels is not None:
            if self.loss_fn == "bce":
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            elif self.loss_fn == "margin":
                loss = MarginLoss()(logits, labels)
            else:
                raise ValueError(f"Unrecognized loss_fn {self.loss_fn}")

        return SmallSuccOutput(
            loss = loss,
            logits = logits,
        )


