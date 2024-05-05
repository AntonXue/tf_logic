import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils.logic_utils import *
from my_datasets.task_datasets import *


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
        self.num_rules = reasoner_dataset.num_rules_range[1] - self.num_attack_tokens
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        tokens = (torch.rand(self.num_rules, 2*self.num_vars) < self.hot_prob).long()
        target = (torch.rand(self.num_vars) < self.hot_prob).long()

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
        reasoner_dataset: Dataset,
        dataset_len: int,
    ):
        # The attack dataset is only used to compute a few things
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.hot_prob = reasoner_dataset.exp_hots / self.num_vars
        self.num_rules = reasoner_dataset.num_rules_range[1] - 1 # We generate one adversarial rule
        self.dataset_len = dataset_len

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

        num_sp_rules = 4

        # Other rules
        other_antes = (torch.rand(self.num_rules-num_sp_rules, self.num_vars) < self.hot_prob).long()
        other_antes[:,e] = 1
        other_conseqs = (torch.rand(self.num_rules-num_sp_rules, self.num_vars) < self.hot_prob).long()
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


