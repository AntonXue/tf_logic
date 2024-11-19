from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision; torchvision.disable_beta_transforms_warning()
from transformers import GPT2Model, GPT2Config
from transformers.utils import ModelOutput


class SelfAttnHead(nn.Module):
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


@dataclass
class TheoryOutput(ModelOutput):
    logits: torch.FloatTensor | None = None
    attn_wts: torch.FloatTensor | None = None


class TheoryModel(nn.Module):
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
        self.attn = SelfAttnHead(embed_dim)
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
        return TheoryOutput(
            logits = x,
            attn_wts = attn_wts,
        )


@dataclass
class AutoregTheoryOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    attn_wts: tuple[torch.FloatTensor] | None = None


class AutoregTheoryModel(nn.Module):
    """ Autoregressive version of the theory model. """
    def __init__(
        self,
        num_props: int,
        num_steps: int,
        embed_dim: int | None = None,
        do_layer_norm: bool = False
    ):
        super().__init__()
        self.num_props = num_props
        self.input_dim = 2 * num_props
        self.embed_dim = 2 * num_props if embed_dim is None else embed_dim
        self.embed_fn = nn.Linear(self.input_dim, self.embed_dim)
        self.cls_head = nn.Linear(self.embed_dim, self.num_props)
        self.model = TheoryModel(self.embed_dim, do_layer_norm=do_layer_norm)
        self.num_steps = num_steps
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ):
        _, _, d = tokens.shape
        all_logits = ()
        all_attn_wts = ()
        all_tokens = tokens

        for t in range(self.num_steps):
            x = self.embed_fn(all_tokens.float())
            out = self.model(x)
            logits = self.cls_head(out.logits[:,-1])
            succ = (logits > 0).long()
            succ_token = torch.cat([torch.zeros_like(succ), succ], dim=-1)
            all_tokens = torch.cat([all_tokens, succ_token.view(-1,1,d)], dim=1).long()

            all_logits += (logits,)
            all_attn_wts += (out.attn_wts,)

        all_logits = torch.stack(all_logits, dim=1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(all_logits, labels.float())

        return AutoregTheoryOutput(
            loss = loss,
            logits = all_logits,
            attn_wts = all_attn_wts,
        )


@dataclass
class AutoregGPT2Output(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    attn_wts: torch.FloatTensor | None = None


class AutoregGPT2Model(nn.Module):
    """ Autoregressive model with a GPT-2 core. """
    def __init__(
        self,
        num_props: int,
        num_steps: int,
        embed_dim: int | None = None,
    ):
        super().__init__()
        self.num_props = num_props
        self.input_dim = 2 * num_props
        self.embed_dim = 2 * num_props if embed_dim is None else embed_dim

        self.embed_fn = nn.Linear(self.input_dim, self.embed_dim)
        self.cls_head = nn.Linear(self.embed_dim, self.num_props)

        # GPT-2 with positional encoding disabled
        self.gpt2 = GPT2Model(GPT2Config(
            n_embd = self.embed_dim,
            n_head = 1,
            n_layer = 1,
        ))
        self.gpt2.wpe.requires_grad_(False)
        self.gpt2.wpe.weight.fill_(0)

        self.num_steps = num_steps
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ):
        _, _, d = tokens.shape
        all_logits = ()
        all_attn_wts = ()
        all_tokens = tokens

        for t in range(self.num_steps):
            x = self.embed_fn(all_tokens.float())
            out = self.gpt2(inputs_embeds=x, output_attentions=True)
            logits = self.cls_head(out.last_hidden_state[:,-1])

            succ = (logits > 0).long()
            succ_token = torch.cat([torch.zeros_like(succ), succ], dim=-1)
            all_tokens = torch.cat([all_tokens, succ_token.view(-1,1,d)], dim=1).long()

            all_logits += (logits,)
            all_attn_wts += out.attentions # Tuple

        all_logits = torch.stack(all_logits, dim=1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(all_logits, labels.float())

        return AutoregGPT2Output(
            logits = all_logits,
            attn_wts = all_attn_wts,
            loss = loss,
        )


