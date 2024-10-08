import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils.logic_utils import *
from my_datasets.task_datasets import *


def hot(i, n):
    return F.one_hot(i, n)


def make_common_stuff(num_vars, num_rules, hot_prob):
    """
    Make rules of the following form that is common to all the experiments
        a -> b
        a -> c
        a -> d
        b,c -> e
        c,d -> f
        e,f -> g
    """
    n, r, p = num_vars, num_rules, hot_prob
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

    # Other rules
    num_sp = special_rules.size(0)
    num_others = r - special_rules.size(0)
    other_antes = (torch.rand(num_others, n) < p).long()
    other_antes[:,h] = 1
    other_conseqs = (torch.rand(num_others, n) < p).long()
    other_rules = torch.cat([other_antes, other_conseqs], dim=-1).long()

    # Gather all the rules
    rules = torch.cat([special_rules, other_rules], dim=0)
    perm = torch.randperm(rules.size(0))
    rules = rules[perm]

    # Append the initial state to the token sequence
    init_token = torch.cat([torch.zeros(n), hot(a,n)])
    tokens = torch.cat([rules, init_token.view(1,2*n)], dim=0)

    return {
        "tokens": tokens,
        "infos": infos,
        "perm": perm
    }


class SuppressRuleDataset(Dataset):
    """
    The goal is to suppress c,d -> f

    Proof states:
        {a} -> {abcd} -> {abcde} -> {abcde}
    """
    def __init__(
        self,
        reasoner_dataset: Dataset,
        num_attack_tokens: int,
        dataset_len: int,
    ):
        # The reasoner dataset is only used to compute a few things
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.hot_prob = reasoner_dataset.exp_hots / self.num_vars
        self.num_rules = reasoner_dataset.num_rules - num_attack_tokens
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n = self.num_vars
        stuff = make_common_stuff(n, self.num_rules, self.hot_prob)
        a, b, c, d, e, f, g, h = stuff["infos"]

        labels = torch.stack([
            hot(a,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n),
        ])

        inv_perm = stuff["perm"].argsort()
        ab_idx, ac_idx, ad_idx, bce_idx, cdf_idx, efg_idx = inv_perm[:6]
        hints = hot(c,n) + hot(d,n) - hot(f,n)

        # Need to calculate the new label that's supposed to happen
        return {
            "tokens": stuff["tokens"],
            "labels": labels,
            "infos": stuff["infos"],
            "ab_index": ab_idx,
            "ac_index": ac_idx,
            "ad_index": ad_idx,
            "bce_index": bce_idx,
            "cdf_index": cdf_idx,
            "efg_index": efg_idx,
            "hints": hints,
        }


class FactAmnesiaDataset(Dataset):
    """
    The goal is to forget "a"

    Proof states:
        {a} -> {bcd} -> {abcdef} -> {bcdefgh}
    """
    def __init__(
        self,
        reasoner_dataset: Dataset,
        num_attack_tokens: int,
        dataset_len: int,
    ):
        # The reasoner dataset is only used to compute a few things
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.hot_prob = reasoner_dataset.exp_hots / self.num_vars
        self.num_rules = reasoner_dataset.num_rules - num_attack_tokens
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n = self.num_vars
        stuff = make_common_stuff(n, self.num_rules, self.hot_prob)
        a, b, c, d, e, f, g, h = stuff["infos"]

        labels = torch.stack([
            hot(a,n),
            hot(b,n) + hot(c,n) + hot(d,n),
            hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n),
            hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n) + hot(g,n),
        ])

        hints = hot(a,n)

        return {
            "tokens": stuff["tokens"],
            "labels": labels,
            "infos": stuff["infos"],
            "hints": hints,
        }


class CoerceStateDataset(Dataset):
    def __init__(
        self,
        reasoner_dataset: Dataset,
        num_attack_tokens: int,
        dataset_len: int,
    ):
        # The reasoner dataset is only used to compute a few things
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.hot_prob = reasoner_dataset.exp_hots / self.num_vars
        self.num_rules = reasoner_dataset.num_rules - num_attack_tokens
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n, p = self.num_vars, self.hot_prob
        stuff = make_common_stuff(self.num_vars, self.num_rules, self.hot_prob)
        infos = stuff["infos"]
        a, b, c, d = infos[0], infos[1], infos[2], infos[3]

        target1 = ((torch.rand(n) < p) + hot(b,n) - hot(a,n) - hot(c,n) - hot(d,n)).clamp(0,1)
        target2 = ((torch.rand(n) < p) + hot(c,n) - hot(a,n) - hot(b,n) - hot(d,n)).clamp(0,1)
        target3 = ((torch.rand(n) < p) + hot(d,n) - hot(a,n) - hot(c,n) - hot(c,n)).clamp(0,1)
        targets = torch.stack([target1, target2, target3]).long()
        labels = torch.cat([hot(a,n).view(1,-1), targets], dim=0).long()
        return {
            "tokens": stuff["tokens"],
            "labels": labels,
            "infos": stuff["infos"],
            "hints": targets,
        }


