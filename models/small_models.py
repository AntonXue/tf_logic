from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from transformers import GPT2Config, GPT2Model

from .common import *
from .loss_functions import *


@dataclass
class SmallSuccOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    others: Optional[dict] = None


class SmallGPT2(nn.Module):
    def __init__(
        self,
        num_vars: int,
        embed_dim: int,
        loss_fn: str = "bce",
        attn_loss_scale: float = 0.0
    ):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim
        self.embed_fn = nn.Linear(2*num_vars, embed_dim, bias=True)
        self.gpt2 = GPT2Model(GPT2Config(n_embd=embed_dim, n_layer=1, n_head=1))

        # Turn off positional encoding
        self.gpt2.wpe.requires_grad_(False)
        self.gpt2.wpe.weight.fill_(0)

        # Turn off layer norms
        """
        self.gpt2.h[0].ln_1 = nn.Identity()
        self.gpt2.h[0].ln_2 = nn.Identity()
        self.gpt2.ln_f = nn.Identity()
        """

        self.cls_head = nn.Linear(embed_dim, num_vars, bias=False)
        self.loss_fn = loss_fn
        self.attn_loss_scale = attn_loss_scale

        if loss_fn == "bce":
            self.lf = nn.BCEWithLogitsLoss()
        elif loss_fn == "margin":
            self.lf = MarginLoss()
        else:
            raise ValueError(f"Unrecognized loss_fn {self.loss_fn}")
        
        self.alf = nn.BCELoss()

    @property
    def desc_str(self):
        alstr = f"als{self.attn_loss_scale:.3f}"
        return f"gpt2_n{self.num_vars}_d{self.embed_dim}_{self.loss_fn}_{alstr}"

    def forward(self, tokens: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        N, L, _ = tokens.shape
        x = self.embed_fn(tokens.float())
        out = self.gpt2(inputs_embeds=x, output_attentions=True)
        logits = self.cls_head(out.last_hidden_state)[:,-1]
        attn_out = out.attentions[0].view(N,L,L)

        loss = None
        if labels is not None:
            loss = self.lf(logits, labels.float())
            antes, conseqs = tokens.chunk(2, dim=-1)
            state = conseqs[:,-1:]
            hits = torch.bmm((state - torch.ones_like(state)), antes.transpose(-1,-2))  # <= 0
            hits = (hits + 1).view(N,L).clamp(0,1)
            attn_loss = self.alf(attn_out[:,-1].clamp(0,1), hits.float())
            loss += self.attn_loss_scale * attn_loss

        return SmallSuccOutput(
            loss = loss,
            logits = logits,
            others = {
                "attn_out": attn_out,
            }
        )



class SmallTfA(nn.Module):
    def __init__(
        self,
        num_vars: int,
        attn_fn: str = "softmax",
        loss_fn: str = "bce",
        init_value: Optional[float] = None
    ):
        super().__init__()
        self.num_vars = num_vars
        self.Wa = nn.Linear(2 * num_vars, 2 * num_vars, bias=True)
        self.Wb = nn.Linear(2 * num_vars, num_vars, bias=True)

        self.attn_fn = attn_fn
        if attn_fn == "softmax":
            self.af = lambda x: F.softmax(x, dim=-1)
        elif attn_fn == "sigmoid":
            self.af = F.sigmoid
        else:
            raise ValueError(f"Unknown attn_fn {attn_fn}")

        self.init_value = init_value
        if init_value is not None:
            self.Wa.weight.data.fill_(init_value)
            self.Wa.bias.data.fill_(init_value)
            self.Wb.weight.data.fill_(init_value)
            self.Wb.bias.data.fill_(init_value)

        self.loss_fn = loss_fn
        if loss_fn == "bce":
            self.lf = nn.BCEWithLogitsLoss()
        elif loss_fn == "margin":
            self.lf = MarginLoss()
        elif loss_fn == "regcat":
            self.lf = RegCatLoss()
        elif loss_fn == "l2":
            self.lf = L2Loss()
        else:
            raise ValueError(f"Unknown loss_fn {loss_fn}")

    @property
    def desc_str(self):
        afstr = self.attn_fn
        ivstr = "IVR" if self.init_value is None else f"IV{self.init_value}"
        lfstr = self.loss_fn
        return f"tfa_n{self.num_vars}_{afstr}_{lfstr}_{ivstr}"

    def forward(self, tokens: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        N, L, _ = tokens.shape
        device = tokens.device
        x = tokens.float()
        wts = self.af(torch.bmm(self.Wa(x), x.transpose(1,2)))
        y = self.Wb(torch.bmm(wts, x))
        logits = y[:,-1]
        loss = None if labels is None else self.lf(logits, labels.float())
        return SmallSuccOutput(
            loss = loss,
            logits = logits,
        )


class SmallTfB(nn.Module):
    def __init__(
        self,
        num_vars: int,
        embed_dim: int,
        attn_fn: str = "softmax",
        loss_fn: str = "bce",
        init_value: Optional[float] = None
    ):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(2 * num_vars, embed_dim, bias=True)
        self.Wk = nn.Linear(2 * num_vars, embed_dim, bias=False)
        self.Wv = nn.Linear(2 * num_vars, num_vars, bias=True)

        self.attn_fn = attn_fn
        if attn_fn == "softmax":
            self.af = lambda x: F.softmax(x, dim=-1)
        elif attn_fn == "sigmoid":
            self.af = F.sigmoid
        else:
            raise ValueError(f"Unknown attn_fn {attn_fn}")

        self.init_value = init_value
        if init_value is not None:
            self.Wa.weight.data.fill_(init_value)
            self.Wa.bias.data.fill_(init_value)
            self.Wb.weight.data.fill_(init_value)
            self.Wb.bias.data.fill_(init_value)

        self.loss_fn = loss_fn
        if loss_fn == "bce":
            self.lf = nn.BCEWithLogitsLoss()
        elif loss_fn == "margin":
            self.lf = MarginLoss()
        elif loss_fn == "regcat":
            self.lf = RegCatLoss()
        else:
            raise ValueError(f"Unknown loss_fn {loss_fn}")

    @property
    def desc_str(self):
        afstr = self.attn_fn
        ivstr = "IVR" if self.init_value is None else f"IV{self.init_value}"
        lfstr = self.loss_fn
        return f"tfb_n{self.num_vars}_d{self.embed_dim}_{afstr}_{lfstr}_{ivstr}"

    def forward(self, tokens: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        N, L, _ = tokens.shape
        device = tokens.device
        x = self.embed_fn(tokens.float())
        
        mask = torch.triu(torch.ones(L,L), diagonal=1).view(1,L,L).repeat(N,1,1).to(device) * -999
        Q, K = self.Wq(x), self.Wk(x)

        wts = F.attn_fn(torch.bmm(Q, K.transpose(1,2)) + mask)
        a = torch.bmm(wts, V)
        y = (x + a) + self.ffwd(x + a)
        logits = self.cls_head(y)[:,-1]
        loss = None if labels is None else self.lf(logits, labels.float())
        return SmallSuccOutput(
            loss = loss,
            logits = logits,
        )



class SmallTfE(nn.Module):
    def __init__(
        self,
        num_vars: int,
        init_value: Optional[float] = None
    ):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim = 2 * num_vars

        self.Wa = nn.Linear(embed_dim, embed_dim, bias=True)
        self.Wb = nn.Linear(embed_dim, num_vars, bias=True)

        self.init_value = init_value
        if init_value is not None:
            self.Wa.weight.data.fill_(init_value)
            self.Wa.bias.data.fill_(init_value)
            self.Wb.weight.data.fill_(init_value)
            self.Wb.bias.data.fill_(init_value)

    @property
    def desc_str(self):
        ivstr = "IVR" if self.init_value is None else f"IV{self.init_value}"
        return f"tfe_n{self.num_vars}_{ivstr}"

    def forward(self, tokens: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        N, L, _ = tokens.shape
        device = tokens.device

        x = tokens.float()
        wts = F.softmax(torch.bmm(self.Wa(x), x.transpose(1,2)), dim=-1)
        a = torch.bmm(wts, x)
        y = self.Wb(a)
        logits = y[:,-1]
        
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())

        return SmallSuccOutput(
            loss = loss,
            logits = logits,
        )

