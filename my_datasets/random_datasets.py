import torch
from torch.utils.data import Dataset
from my_datasets.dataset_utils import *

from . import logic

class OneShotEmbedsDataset(Dataset):
    """ For task of checking one-shot QED, generate a bunch of random rules """
    def __init__(
        self,
        num_rules: int,
        num_vars: int,
        ante_prob: float,
        conseq_prob: float,
        theorem_prob: float,
        dataset_len: int,
        ensure_facts: bool = True,
        seed: int = 1234
    ):
        self.num_rules = num_rules
        self.num_vars = num_vars
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.theorem_prob = theorem_prob
        self.dataset_len = dataset_len
        self.ensure_facts = ensure_facts
        self.seed = seed

    def __len__(self):
        """ We can indefinitely generate more, but set an artificial cap """
        return self.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)  # How to guarantee determinism
        rules = logic.random_rules(
            batch_size = 1,
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            ensure_facts = self.ensure_facts)

        thm = (torch.rand(1, self.num_vars) < self.theorem_prob).long()
        qed = logic.prove_theorem(rules, thm)["qed"]
        return {
            "rules" : rules[0],
            "theorem" : thm[0],
            "labels" : qed[0]
        }

class OneShotTextDataset(Dataset):
    """ For task of checking one-shot QED, generate a bunch of random rules 
    The rules and theorem are represented as strings"""
    def __init__(
        self,
        num_rules: int,
        num_vars: int,
        ante_prob: float,
        conseq_prob: float,
        theorem_prob: float,
        dataset_len: int,
        ensure_facts: bool = True,
        seed: int = 1234,
        tokenizer: object = None
    ):
        self.num_rules = num_rules
        self.num_vars = num_vars
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.theorem_prob = theorem_prob
        self.dataset_len = dataset_len
        self.ensure_facts = ensure_facts
        self.seed = seed
        self.tokenizer = tokenizer

    def __len__(self):
        """ We can indefinitely generate more, but set an artificial cap """
        return self.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)  # How to guarantee determinism
        rules = logic.random_rules(
            batch_size = 1,
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            ensure_facts = self.ensure_facts)

        thm = (torch.rand(1, self.num_vars) < self.theorem_prob).long()
        qed = logic.prove_theorem(rules, thm)["qed"]

        entry = {
                "rules" : rules[0],
                "theorem" : thm[0],
                "labels" : qed[0]
            }
        
        entry_str = get_string_rep(entry)
        if not self.tokenizer:
            return Exception("Tokenizer not provided.")
        
        encoding = self.tokenizer(entry_str, truncation=True)
        return {
            "data": entry_str,
            "label": qed[0],
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask
        }


class NextStateEmbedsDataset(Dataset):
    """ For task of generating the next state, generate a bunch of random rules """
    def __init__(
        self,
        num_rules: int,
        num_vars: int,
        ante_prob: float,
        conseq_prob: float,
        state_prob: float,
        dataset_len: int,
        ensure_facts: bool = True,
        seed: int = 1234
    ):
        self.num_rules = num_rules
        self.num_vars = num_vars
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.state_prob = state_prob
        self.dataset_len = dataset_len
        self.ensure_facts = ensure_facts
        self.seed = seed

    def __len__(self):
        """ We can indefinitely generate more, but set an artificial cap """
        return self.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)
        rules = logic.random_rules(
            batch_size = 1,
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            ensure_facts = self.ensure_facts)

        state = (torch.rand(1, self.num_vars) < self.state_prob).long()
        succ, _ = logic.step_rules(rules, state)

        return {
            "rules" : rules[0],
            "state" : state[0],
            "labels" : succ[0]
        }


class NextStateTextDataset(Dataset):
    pass


class AutoRegKStepsEmbedsDataset(Dataset):
    def __init__(
        self,
        num_rules: int,
        num_vars: int,
        num_steps: int,
        ante_prob: float,
        conseq_prob: float,
        state_prob: float,
        dataset_len: int,
        ensure_facts: bool = True,
        seed: int = 1234
    ):
        self.num_rules = num_rules
        self.num_vars = num_vars
        self.num_steps = num_steps
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.state_prob = state_prob
        self.dataset_len = dataset_len
        self.ensure_facts = ensure_facts
        self.seed = seed

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)
        rules = logic.random_rules(
            batch_size = 1,
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            ensure_facts = self.ensure_facts)

        init_state = (torch.rand(1, self.num_vars) < self.state_prob).long()
        tmp = init_state
        succs = ()
        for t in range(self.num_steps):
            tmp, _ = logic.step_rules(rules, tmp)
            succs = succs + (tmp,)

        succs = torch.cat(succs, dim=0).long()

        return {
            "rules": rules[0],
            "state": init_state[0],
            "labels" : succs
        }


class AutoRegKStepsTextDataset(Dataset):
    pass


