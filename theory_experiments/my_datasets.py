import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def hot(i,n):
    return F.one_hot(i,n)


class AutoregCustomTokensDataset(Dataset):
    """
        a -> b
        a -> c
        a -> d
        b,c -> e
        c,d -> f
        e,f -> g

        Proof states: {a} -> {abcd} -> {abcdef} -> {abcdefg}
    """
    def __init__(
        self,
        num_props: int,
        dataset_len: int,
        num_rules: int = 32,
        exp_hots: float = 3.0,
    ):
        assert num_props >= 8
        self.num_props = num_props
        self.num_rules = num_rules
        self.exp_hots = exp_hots
        self.hot_prob = exp_hots / num_props
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n, r, p = self.num_props, self.num_rules, self.hot_prob
        infos = torch.randperm(n)[:8]
        a, b, c, d, e, f, g, h = infos

        special_rules = torch.stack([
            torch.cat([hot(a,n), hot(b,n)]), # a -> b
            torch.cat([hot(a,n), hot(c,n)]), # a -> c
            torch.cat([hot(a,n), hot(d,n)]), # a -> d
            torch.cat([hot(b,n) + hot(c,n), hot(e,n)]), # b,c -> e
            torch.cat([hot(c,n) + hot(d,n), hot(f,n)]), # c,d -> f
            torch.cat([hot(e,n) + hot(f,n), hot(g,n)]), # e,f -> g
        ]).long()

        num_others = r - special_rules.size(0)
        other_antes = (torch.rand(num_others, n) < p).long()
        other_antes[:,h] = 1
        other_conseqs = (torch.rand(num_others, n) < p).long()
        other_rules = torch.cat([other_antes, other_conseqs], dim=-1)
        rules = torch.cat([special_rules, other_rules], dim=0)
        rules = rules[torch.randperm(rules.size(0))]

        init_token = torch.cat([torch.zeros(n), hot(a,n)])
        tokens = torch.cat([rules, init_token.view(1,2*n)], dim=0)

        labels = torch.stack([
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n) + hot(g,n)
        ]).long()

        return {
            "tokens": tokens,
            "labels": labels,
            "infos": infos
        }

