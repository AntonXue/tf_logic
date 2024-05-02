import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils.logic_utils import *
from my_datasets.task_datasets import *


class BadSuffixDataset(Dataset):
    """ Dataset for performing attacks on models trained with AutoregKSteps """
    def __init__(
        self,
        reasoner_dataset: Dataset,
        num_attack_tokens: int,
        dataset_len: Optional[int] = None,
    ):
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.num_attack_tokens = num_attack_tokens
        self.dataset_len = len(reasoner_dataset) if dataset_len is None else dataset_len
        assert self.dataset_len <= len(reasoner_dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        tokens = self.reasoner_dataset[idx]["tokens"]
        tokens = tokens[self.num_attack_tokens:]    # Slice off the start to stay in-distribution wrt length
        target = (torch.rand(self.num_vars) < 0.5).long()

        # We have to use huggingface trainer's naming convention for "labels"
        return {
            "tokens": tokens,
            "labels": target
        }


class SuppressRuleDataset(Dataset):
    """
    This is a diamond-shaped dataset. We construct a dataset that has rules of form

    Rules:
        _ -> a
        a -> b
        a -> c
        b,c -> d
        ...

    The goal for the attacker is to generate a token that suppresses the rule a->b
    """
    def __init__(
        self,
        num_vars: int,
        num_rules: int,
        dataset_len: int,
        hot_prob: Optional[float] = None
    ):
        assert num_vars > 4 and num_rules > 5
        self.num_vars = num_vars
        self.num_rules = num_rules
        self.dataset_len = dataset_len
        self.hot_prob = (1.0/num_vars) if hot_prob is None else hot_prob

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n = self.num_vars
        abcde = torch.randperm(n)[:5]
        a, b, c, d, e = abcde

        special_rules = torch.stack([
            torch.cat([torch.zeros(n), F.one_hot(a,n)]), # _ -> a
            torch.cat([F.one_hot(a,n), F.one_hot(b,n)]), # a -> b
            torch.cat([F.one_hot(a,n), F.one_hot(c,n)]), # a -> c
            torch.cat([F.one_hot(b,n) + F.one_hot(c,n), F.one_hot(d,n)]) # b,c -> d
        ]).long()

        # Other rules
        other_antes = torch.rand(self.num_rules-4, self.num_vars) < self.hot_prob
        other_antes[:,e] = 1 # This guarantees that the other rules are never triggered
        other_conseqs = torch.rand(self.num_rules-4, self.num_vars) < self.hot_prob
        other_rules = torch.cat([other_antes, other_conseqs], dim=-1).long()

        # Gather all the rules
        rules = torch.cat([special_rules, other_rules], dim=0)
        rules = rules[torch.randperm(rules.size(0))]

        # Append the initial state to the token sequence
        tokens = torch.cat([rules, torch.zeros(1,2*n)], dim=0)

        # Labels for a three-step process
        labels = torch.stack([
            F.one_hot(a,n),
            F.one_hot(a,n) + F.one_hot(c,n),
            F.one_hot(a,n) + F.one_hot(c,n),
        ]).long()

        # Need to calculate the new label that's supposed to happen
        return {
            "tokens": tokens,
            "labels": labels,
            "abcde": abcde
        }


