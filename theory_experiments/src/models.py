from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput


@dataclass
class MyTheoryOutput(ModelOutput):
    logits: torch.FloatTensor | None = None
    attn_outs: torch.FloatTensor | None = None
    attn_wts: torch.FloatTensor | None = None


class MySelfAttnHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.FloatTensor):
        """ x: (batch_size, seq_len, embed_dim) """
        B, L, d = x.shape
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        mask = torch.triu(torch.ones(L,L), diagonal=1).view(1,L,L).repeat(B,1,1).to(x.device) * -999
        wts = F.softmax((torch.bmm(Q, K.transpose(1,2)) / (d**0.5)) + mask, dim=2) # (B,L,L)
        out = torch.bmm(wts, V)
        return out, wts


class MyTheoryModel(nn.Module):
    """ A one-layer, one-head transformer model of some embedding dimension. """
    def __init__(
        self,
        embed_dim: int,
        do_layer_norm: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim) if do_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim) if do_layer_norm else nn.Identity()
        self.attn = MySelfAttnHead(embed_dim)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim)
        )

    def forward(self, x: torch.FloatTensor):
        """ x: (batch_size, seq_len, embed_dim) """
        attn_out, attn_wts = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ffwd(self.norm2(x))
        return MyTheoryOutput(
            logits = x,
            attn_outs = attn_out,
            attn_wts = attn_wts,
        )


@dataclass
class MyAutoregTheoryOutput(ModelOutput):
    logits: torch.FloatTensor | None = None
    attn_outs: tuple[torch.FloatTensor] | None = None
    attn_wts: tuple[torch.FloatTensor] | None = None
    loss: torch.FloatTensor | None = None


class MyAutoregTheoryModel(nn.Module):
    """ Autoregressive version of the theory model. """
    def __init__(
        self,
        num_props: int,
        num_steps: int,
        do_layer_norm: bool = False
    ):
        super().__init__()
        self.num_props = num_props
        self.model = MyTheoryModel(2*num_props, do_layer_norm=do_layer_norm)
        self.num_steps = num_steps
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ):
        _, _, d = tokens.shape
        all_logits = ()
        all_attn_outs = ()
        all_attn_wts = ()
        all_tokens = tokens

        for t in range(self.num_steps):
            out = self.model(all_tokens.float())
            logits = out.logits[:,-1,-self.num_props:]
            succ = (logits > 0).long()
            succ_token = torch.cat([torch.zeros_like(succ), succ], dim=-1)
            all_tokens = torch.cat([all_tokens, succ_token.view(-1,1,d)], dim=1).long()

            all_logits += (logits,)
            all_attn_outs += (out.attn_outs,)
            all_attn_wts += (out.attn_wts,)

        all_logits = torch.stack(all_logits, dim=1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(all_logits, labels.float())

        return MyAutoregTheoryOutput(
            logits = all_logits,
            attn_outs = all_attn_outs,
            attn_wts = all_attn_wts,
            loss = loss,
        )

