import torch
import torch.nn as nn
import torch.nn.functional as F


str_to_activ_module = {
    "relu" : nn.ReLU(),
    "gelu" : nn.GELU(),
}

def _get_activ_module(activ):
    return str_to_activ_module[activ]


def make_feedforward(in_dim, out_dim, width, depth, activ="relu"):
    """ Make a simple MLP with #depth activations
    """
    assert depth > 0
    modules = [nn.Linear(in_dim, width), _get_activ_module(activ)]
    for i in range(depth-1):
        modules.append(nn.Linear(width, width))
        modules.append(_get_activ_module(activ))
    modules.append(nn.Linear(width, out_dim))
    return nn.Sequential(*modules)


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
    def __init__(self, dim, width, depth, num_heads=2,
                 activ = "relu",
                 do_norm = True,
                 do_norm_first = False,
                 layer_norm_eps = 1e-5,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.width = width
        self.depth = depth
        self.activ = activ

        self.do_norm = do_norm
        self.do_norm_first = do_norm_first

        if do_norm:
            self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.attn = MultiAttention(dim, num_heads)
        self.ffwd = make_feedforward(dim, dim, width, depth, activ=activ)

    def forward(self, x):
        z = x
        if self.do_norm_first:
            z = z + self.attn(self.norm1(z))
            z = z + self.ffwd(self.norm2(z))
        else:
            z = self.norm1(z + self.attn(z))
            z = self.norm2(z + self.ffwd(z))
        return z



