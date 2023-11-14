import torch
from torch.utils.data import Dataset

import logic.prop_logic as proplog

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
        rules = proplog.random_rules(
            batch_size = 1,
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            ensure_facts = self.ensure_facts)

        thm = (torch.rand(1, self.num_vars) < self.theorem_prob).long()
        qed = proplog.prove_theorem(rules, thm)["qed"]
        return {
            "rules" : rules[0],
            "theorem" : thm[0],
            "labels" : qed[0]
        }

class OneShotTextDataset(Dataset):
    pass


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
        rules = proplog.random_rules(
            batch_size = 1,
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            ensure_facts = self.ensure_facts)

        state = (torch.rand(1, self.num_vars) < self.state_prob).long()
        succ, _ = proplog.step_rules(rules, state)

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
        rules = proplog.random_rules(
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
            tmp, _ = proplog.step_rules(rules, tmp)
            succs = succs + (tmp,)

        succs = torch.cat(succs, dim=0).long()

        return {
            "rules": rules[0],
            "state": init_state[0],
            "labels" : succs
        }


class AutoRegKStepsTextDataset(Dataset):
    pass


