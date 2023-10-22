import math
import torch
import torch.nn as nn


class MultiAttention(torch.nn.Module):
    """ Multiheaded attention without splitting across the model_dim
    """
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.attn_heads = nn.ModuleList([nn.MultiheadAttention(model_dim, 1)
                                            for _ in range(num_heads)])

    def forward(self, x, verbose=False):
        outs = [head(x,x,x) for head in self.attn_heads]
        A = torch.stack([out[0] for out in outs])   # Attention output
        W = torch.stack([out[1] for out in outs])   # Attention weights
        if verbose:
            return A.sum(dim=0), W.sum(dim=0)
        else:
            return A.sum(dim=0)


class AFBlock(nn.Module):
    """ A single attention-feedforward block
    """
    def __init__(self, model_dim, width, depth, num_heads,
                 do_norm = True,
                 layer_norm_eps = 1e-5):
        super().__init__()
        # Norm layers
        if do_norm:
            self.norm1 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Attention block
        self.attn = MultiAttention(model_dim, num_heads)

        # Feedforward block
        ffwd_parts = [nn.Linear(model_dim, width), nn.ReLU()]
        for i in range(depth-1):
            ffwd_parts.append(nn.Linear(width, width))
            ffwd_parts.append(nn.ReLU())
        ffwd_parts.append(nn.Linear(width, model_dim))
        self.ffwd = nn.Sequential(*ffwd_parts)

    #
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffwd(x))
        return x


class MyTransformer(nn.Module):
    """ A transformer is consisted of num_blocks number of AFBlocks
    """
    def __init__(self, model_dim, ffwd_width, ffwd_depth, num_heads, num_blocks, **kwargs):
        super().__init__()
        self.model_dim = model_dim
        self.ffwd_width = ffwd_width
        self.ffwd_depth = ffwd_depth
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.attn_ffwds = nn.ModuleList([AFBlock(model_dim = model_dim,
                                                 width = ffwd_width,
                                                 depth = ffwd_depth,
                                                 num_heads = num_heads,
                                                 **kwargs)
                                            for _ in range(num_blocks)])

    #
    def forward(self, x):
        for block in self.attn_ffwds:
            x = block(x)
        return x


class ProverEncoder(nn.Module):
    """ Stolen from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, num_vars, num_rules, model_dim, seq_len,
                 do_norm = True,
                 layer_norm_eps = 1e-5):
        super().__init__()
        self.num_vars = num_vars
        self.num_rules = num_rules
        self.model_dim = model_dim

        assert seq_len > num_rules + 1
        self.seq_len = seq_len

        if do_norm:
            self.norm = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        else:
            self.norm = nn.Identity()

        self.ffwd = nn.Sequential(
                nn.Linear(num_vars * 2, model_dim * 2),
                nn.ReLU(),
                nn.Linear(model_dim * 2, model_dim * 2),
                nn.ReLU(),
                nn.Linear(model_dim * 2, model_dim))

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(seq_len, 1, model_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        

    def forward(self, rules, theorem):
        N, r, n2 = rules.shape
        _, n = theorem.shape
        x = torch.cat([-2*torch.ones_like(theorem), theorem]).view(N,1,n2)
        x = torch.cat([rules, x], dim=1)
        x = x.permute(1,0,2) # (r+1,N,2n)
        x_pad = torch.zeros(self.seq_len - x.size(0), N, n2, device=x.device)
        x = torch.cat([x, x_pad], dim=0)
        x = self.ffwd(x)
        x = x + self.pe[:x.size(0)] # Inject positional information
        x = self.norm(x)
        return x # (seq_len, N, model_dim)


class QedDecoder(nn.Module):
    """ Returns a single bit indicating whether a QED is achieved
    """
    def __init__(self, model_dim, apply_sigmoid=True):
        super().__init__()
        self.model_dim = model_dim
        self.apply_sigmoid = apply_sigmoid
        self.ffwd = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, 1))

    def forward(self, x):   # x : (seq_len, N, model_dim)
        x = self.ffwd(x)
        x = x[-1,:,0]
        if self.apply_sigmoid:
            x = x.sigmoid()
        return x


class StateDecoder(nn.Module):
    """ Returns the state after some whatever steps
    """
    def __init__(self, model_dim, num_vars, apply_sigmoid=True):
        super().__init__()
        self.model_dim = model_dim
        self.num_vars = num_vars
        self.apply_sigmoid = apply_sigmoid
        self.ffwd = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, num_vars))

    def forward(self, x):   # x : (seq_len, N, model_dim)
        x = self.ffwd(x)
        x = x[-1,:,:]
        if self.apply_sigmoid:
            x = x.sigmoid()
        return x


class LogicTransformer(nn.Module):
    """ Put the above together
    """

    def __init__(self, encoder, transformer, decoder):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder

    def forward(self, rules, theorem):
        x = self.encoder(rules, theorem)
        x = self.transformer(x)
        x = self.decoder(x)
        return x


