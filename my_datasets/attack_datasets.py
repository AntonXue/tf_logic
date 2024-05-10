import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils.logic_utils import *
from my_datasets.task_datasets import *


def hot(i, n):
    return F.one_hot(i, n)


class CoerceStateDataset(Dataset):
    """ Dataset for performing attacks on models trained with AutoregKSteps """
    def __init__(
        self,
        reasoner_dataset: Dataset,
        num_attack_tokens: int,
        dataset_len: int,
    ):
        # The attack dataset is only used to compute a few things
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.num_attack_tokens = num_attack_tokens
        self.hot_prob = reasoner_dataset.exp_hots / self.num_vars

        if hasattr(reasoner_dataset, "num_rules_range"):
            self.num_rules = reasoner_dataset.num_rules_range[1] - self.num_attack_tokens
        else:
            self.num_rules = reasoner_dataset.num_rules - self.num_attack_tokens

        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n, r, p = self.num_vars, self.num_rules, self.hot_prob
        tokens = (torch.rand(r, 2*n) < p).long()
        target = (torch.rand(n) < p).long()

        # We have to use huggingface trainer's naming convention for "labels"
        return {
            "tokens": tokens,
            "labels": target
        }


class SuppressRuleDataset(Dataset):
    """
    This is a diamond-shaped dataset. We construct a dataset that has rules of form

    Rules:
        a -> b
        a -> c
        a -> d
        b,c -> e
        c,d -> f
        e,f -> g

    The goal is to suppress a -> b

    Proof states:
        {a} -> {acd} -> {acdf} -> {acdf}
    """
    def __init__(
        self,
        reasoner_dataset: Dataset,
        dataset_len: int,
    ):
        # The attack dataset is only used to compute a few things
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.hot_prob = reasoner_dataset.exp_hots / self.num_vars

        if hasattr(reasoner_dataset, "num_rules_range"):
            self.num_rules = reasoner_dataset.num_rules_range[1] - 1
        else:
            self.num_rules = reasoner_dataset.num_rules - 1

        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n, r, p = self.num_vars, self.num_rules, self.hot_prob
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
        other_antes = (torch.rand(r-special_rules.size(0), n) < p).long()
        other_antes[:,h] = 1
        other_conseqs = (torch.rand(r-special_rules.size(0), n) < p).long()
        other_rules = torch.cat([other_antes, other_conseqs], dim=-1).long()

        # Gather all the rules
        rules = torch.cat([special_rules, other_rules], dim=0)
        perm = torch.randperm(rules.size(0))
        rules = rules[perm]

        # Append the initial state to the token sequence
        init_token = torch.cat([torch.zeros(n), hot(a,n)])
        tokens = torch.cat([rules, init_token.view(1,2*n)], dim=0)

        # Labels for a three-step process
        labels = torch.stack([
            hot(a,n) + hot(c,n) + hot(d,n),
            hot(a,n) + hot(c,n) + hot(d,n) + hot(f,n),
            hot(a,n) + hot(c,n) + hot(d,n) + hot(f,n),
        ])

        # Need to calculate the new label that's supposed to happen
        return {
            "tokens": tokens,
            "labels": labels,
            "infos": infos,
            # Invert the perm with argsort to find where the 0th rule (a->b) is
            "supp_idx": perm.argsort()[0]
        }


class KnowledgeAmnesiaDataset(Dataset):
    """
    Rules:
        a -> b
        a -> c
        a -> d
        b,c -> e
        c,d -> f
        e,f -> g

    The goal is to forget "a"

    Proof states:
        {a} -> {bcd} -> {abcdef} -> {bcdefgh}
    """
    def __init__(
        self,
        reasoner_dataset: Dataset,
        dataset_len: int,
    ):
        self.suppress_rule_dataset = SuppressRuleDataset(
            reasoner_dataset = reasoner_dataset,
            dataset_len = dataset_len
        )
        self.num_vars = self.suppress_rule_dataset.num_vars

    def __len__(self):
        return len(self.suppress_rule_dataset)

    def __getitem__(self, idx):
        item = self.suppress_rule_dataset[idx]
        infos = item["infos"]
        a, b, c, d, e, f, g, h = infos
        n = self.num_vars

        # Labels for a one-step process
        labels = torch.stack([
                       hot(b,n) + hot(c,n) + hot(d,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n),
                       hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n) + hot(g,n),
        ])

        return {
            "tokens": item["tokens"],
            "labels": labels,
            "infos": infos,
        }


