import math
import torch
import torch.nn as nn


class MultiAttention(torch.nn.Module):
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
        z = x
        z = self.norm1(z + self.attn(z))
        z = self.norm2(z + self.ffwd(z))
        return z


class MyTransformer(nn.Module):
    def __init__(self, model_dim, ffwd_width, ffwd_depth, num_heads, num_blocks, **kwargs):
        super().__init__()
        self.model_dim = model_dim
        self.ffwd_width = ffwd_width
        self.ffwd_depth = ffwd_depth
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        blocks = []
        for _ in range(num_blocks):
            blocks.append(AFBlock(model_dim = model_dim,
                                  width = ffwd_width,
                                  depth = ffwd_depth,
                                  num_heads = num_heads,
                                  **kwargs))
        self.attn_ffwds = nn.Sequential(*blocks)


    #
    def forward(self, x):
        return self.attn_ffwds(x)


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
                nn.Linear(num_vars * 2, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim),
        )

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(seq_len, 1, model_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        

    def forward(self, rules, theorem):
        N, r, n2 = rules.shape
        _, n = theorem.shape
        z0 = torch.cat([-1*torch.ones_like(theorem), theorem], dim=1)
        x = torch.cat([rules, z0.view(N,1,n2)], dim=1) # (N,r+1,2n)
        x = x.permute(1,0,2) # (r+1,N,2n)
        x_pad = torch.zeros(self.seq_len - x.size(0), N, n2, device=x.device)
        x = torch.cat([x, x_pad], dim=0)
        x = self.ffwd(x)
        x = x + self.pe[:x.size(0)] # Inject positional information
        x = self.norm(x)
        return x # (seq_len, N, model_dim)


class CheckQedDecoder(nn.Module):
    def __init__(self, model_dim, apply_sigmoid=True):
        super().__init__()
        self.model_dim = model_dim
        self.apply_sigmoid = apply_sigmoid
        self.ffwd = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1))

    def forward(self, x):
        """ x : (seq_len, batch_size, d)
        """
        _, N, _ = x.shape
        z = self.ffwd(x)
        z = z[-1,:,:]
        if self.apply_sigmoid:
            z = z.sigmoid()
        return z.view(N)


# Put the above together
class LogicTransformer(nn.Module):
    def __init__(self, encoder, transformer, decoder):
        super().__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder

    def forward(self, rules, theorem):
        x = self.encoder(rules, theorem)
        y = self.transformer(x)
        qed = self.decoder(y)
        return qed


