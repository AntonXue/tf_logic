import torch
import torch.nn as nn
import torch.nn.functional as F


str_to_activ_module = {
    "relu" : nn.ReLU(),
    "gelu" : nn.GELU(),
}

def _get_activ_module(activ):
    return str_to_activ_module[activ]


class MultiAttention(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn_heads = [nn.MultiheadAttention(dim, 1) for _ in range(num_heads)]

    def forward(self, x, verbose=False):
        outs = [head(x,x,x) for head in self.attn_heads]
        A = torch.stack([out[0] for out in outs])   # Attention output
        W = torch.stack([out[1] for out in outs])   # Attention weights
        if verbose:
            return A.sum(dim=0), W.sum(dim=0)
        else:
            return A.sum(dim=0)


class AttentionFeedforward(nn.Module):
    def __init__(self, embed_dim, width, depth, num_heads,
                 activ = "relu",
                 do_norm = True,
                 layer_norm_eps = 1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.width = width
        self.depth = depth
        self.activ = activ
        self.do_norm = do_norm

        # Norm layers
        if do_norm:
            self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Attention block
        self.attn = MultiAttention(embed_dim, num_heads)

        # Feedforward block
        lins = [nn.Linear(embed_dim, width), _get_activ_module(activ)]
        for i in range(depth-1):
            lins.append(nn.Linear(width, width))
            lins.append(_get_activ_module(activ))
        lins.append(nn.Linear(width, embed_dim))
        self.ffwd = nn.Sequential(*lins)

    #
    def forward(self, x):
        z = x
        z = self.norm1(z + self.attn(z))
        z = self.norm2(z + self.ffwd(z))
        return z


class MyTransformer(nn.Module):
    def __init__(self, embed_dim, ffwd_width, ffwd_depth, num_heads, stack_depth, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.ffwd_width = ffwd_width
        self.ffwd_depth = ffwd_depth
        self.num_heads = num_heads
        self.stack_depth = stack_depth

        self.attn_ffwds = nn.Sequential(*[
                AttentionFeedforward(embed_dim = embed_dim,
                                     width = ffwd_width,
                                     depth = ffwd_depth,
                                     num_heads = num_heads)
                for _ in range(stack_depth)])

    #
    def forward(self, x):
        return self.attn_ffwds(x)



