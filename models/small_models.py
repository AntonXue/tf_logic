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

    def forward(self, logits, labels):
        labels = 2 * labels - 1
        loss = F.relu(self.delta - logits.float() * labels.float()).sum(dim=1).mean()
        return loss

class RegCatLoss(nn.Module):
    def __init__(self, rho: float = None):
        super().__init__()
        self.rho = rho

    def forward(self, logits, labels):
        labels = 2 * labels - 1
        loss = -(logits.float() * labels.float()).sum(dim=1).mean()

        _, n = logits.shape
        rho = (1 / (2*n)) if self.rho is None else rho
        reg = rho * (torch.norm(logits, p=2, dim=1) ** 2).mean()
        return loss + reg

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
        return f"tfa_n{self.num_vars}_d{self.embed_dim}_{self.loss_fn}"

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
            elif self.loss_fn == "regcat":
                loss = RegCatLoss()(logits, labels)
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
        return f"tfb_n{self.num_vars}_{self.loss_fn}"

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
            elif self.loss_fn == "regcat":
                loss = RegCatLoss()(logits, labels)
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
        loss_fn: str = "bce",
        init_value: Optional[float] = None,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim = 2 * num_vars + 1
        self.Wa = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wb = nn.Linear(embed_dim, num_vars, bias=True)
        self.loss_fn = loss_fn
        
        self.attn_fn_str = attn_fn
        if attn_fn == "relu":
            self.attn_fn = F.relu
        elif attn_fn == "sigmoid":
            self.attn_fn = F.sigmoid
        elif attn_fn == "softplus":
            self.attn_fn = F.softplus
        elif attn_fn == "softmax":
            self.attn_fn = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unrecognized attn_fn {attn_fn}")

        self.init_value = init_value
        if init_value is not None:
            self.Wa.weight.data.fill_(init_value)
            if self.Wa.bias is not None:
                self.Wa.bias.data.fill_(init_value)

            self.Wb.weight.data.fill_(init_value)
            if self.Wb.bias is not None:
                self.Wb.bias.data.fill_(init_value)

    @property
    def desc_str(self):
        ivstr = "IvR" if self.init_value is None else f"Iv{self.init_value}"
        return f"tfc_{self.attn_fn_str}_n{self.num_vars}_{self.loss_fn}_{ivstr}"

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
            elif self.loss_fn == "regcat":
                loss = RegCatLoss()(logits, labels)
            else:
                raise ValueError(f"Unrecognized loss_fn {self.loss_fn}")

        return SmallSuccOutput(
            loss = loss,
            logits = logits,
        )



class SmallTfD(nn.Module):
    def __init__(
        self,
        num_vars: int,
        attn_fn: str = "relu",
        loss_fn: str = "bce",
        init_value: Optional[float] = None,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim = 2 * num_vars + 1
        
        self.Wa = nn.Linear(embed_dim, embed_dim, bias=False)
        self.c = nn.Parameter(torch.randn(()))

        self.loss_fn = loss_fn
        
        self.attn_fn_str = attn_fn
        if attn_fn == "relu":
            self.attn_fn = F.relu
        elif attn_fn == "sigmoid":
            self.attn_fn = F.sigmoid
        elif attn_fn == "softplus":
            self.attn_fn = F.softplus
        elif attn_fn == "softmax":
            self.attn_fn = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unrecognized attn_fn {attn_fn}")

        self.init_value = init_value
        if init_value is not None:
            self.Wa.weight.data.fill_(init_value)
            self.c.data.fill_(init_value)

    @property
    def desc_str(self):
        ivstr = "IvR" if self.init_value is None else f"Iv{self.init_value}"
        return f"tfd_{self.attn_fn_str}_n{self.num_vars}_{self.loss_fn}_{ivstr}"

    def forward(self, tokens: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        N, L, _ = tokens.shape
        device = tokens.device

        x = torch.cat([torch.ones(N,L,1).to(device), tokens], dim=2).float()
        wts = self.attn_fn(torch.bmm(self.Wa(x), x.transpose(1,2)))
        a = torch.bmm(wts, x)
        y = a[:,:,-self.num_vars:] + self.c
        logits = y[:,-1]

        loss = None
        if labels is not None:
            if self.loss_fn == "bce":
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            elif self.loss_fn == "margin":
                loss = MarginLoss()(logits, labels)
            elif self.loss_fn == "regcat":
                loss = RegCatLoss()(logits, labels)
            else:
                raise ValueError(f"Unrecognized loss_fn {self.loss_fn}")

        return SmallSuccOutput(
            loss = loss,
            logits = logits,
        )



