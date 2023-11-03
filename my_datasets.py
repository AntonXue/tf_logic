from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import logic.prop_logic as proplog

""" One step QED dataset """

@dataclass
class OneShotQedDatasetConfig:
    num_rules: int
    num_vars: int
    ante_prob: float
    conseq_prob: float
    theorem_prob: float
    dataset_len: int
    ensure_facts: bool = True
    seed: int = 1234

class OneShotQedDataset(Dataset):
    """ For task of checking one-shot QED, generate a bunch of random rules """
    def __init__(self, config: OneShotQedDatasetConfig):
        """ Some rules """
        self.config = config

    def __len__(self):
        """ We can indefinitely generate more, but set an artificial cap """
        return self.config.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.config.seed + idx)  # How to guarantee determinism
        rules = proplog.random_rules(
                batch_size = 1,
                num_rules = self.config.num_rules,
                num_vars = self.config.num_vars,
                ante_prob = self.config.ante_prob,
                conseq_prob = self.config.conseq_prob,
                ensure_facts = self.config.ensure_facts)

        thm = (torch.rand(1, self.config.num_vars) < self.config.theorem_prob).long()
        qed = proplog.prove_theorem(rules, thm)["qed"]

        return {
            "rules" : rules[0],
            "theorem" : thm[0],
            "labels" : qed[0]
        }


""" Checking succ """

@dataclass
class PredictSuccDatasetConfig:
    num_rules: int
    num_vars: int
    ante_prob: float
    conseq_prob: float
    state_prob: float
    dataset_len: int
    ensure_facts: bool = True
    seed: int = 1234


class PredictSuccDataset(Dataset):
    """ For task of generating the next state, generate a bunch of random rules """
    def __init__(self, config: PredictSuccDatasetConfig):
        """ Some rules """
        self.config = config

    def __len__(self):
        """ We can indefinitely generate more, but set an artificial cap """
        return self.config.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.config.seed + idx)
        rules = proplog.random_rules(
                batch_size = 1,
                num_rules = self.config.num_rules,
                num_vars = self.config.num_vars,
                ante_prob = self.config.ante_prob,
                conseq_prob = self.config.conseq_prob,
                ensure_facts = self.config.ensure_facts)

        state = (torch.rand(1, self.config.num_vars) < self.config.state_prob).long()
        succ, _ = proplog.step_rules(rules, state)

        return {
            "rules" : rules[0],
            "state" : state[0],
            "labels" : succ[0]
        }

""" Auto-regressive fixed steps """

@dataclass
class AutoRegFixedStepsDatasetConfig:
    num_rules: int
    num_vars: int
    ante_prob: float
    conseq_prob: float
    state_prob: float
    num_steps: int
    dataset_len: int
    ensure_facts: bool = True
    seed: int = 1234

class AutoRegFixedStepsDataset(Dataset):
    def __init__(self, config: AutoRegFixedStepsDatasetConfig):
        self.config = config

    def __len__(self):
        return self.config.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.config.seed + idx)
        rules = proplog.random_rules(
                batch_size = 1,
                num_rules = self.config.num_rules,
                num_vars = self.config.num_vars,
                ante_prob = self.config.ante_prob,
                conseq_prob = self.config.conseq_prob,
                ensure_facts = self.config.ensure_facts)

        init_state = (torch.rand(1, self.config.num_vars) < self.config.state_prob).long()
        tmp = init_state
        succs = ()
        for t in range(self.config.num_steps):
            tmp, _ = proplog.step_rules(rules, tmp)
            succs = succs + (tmp,)

        succs = torch.cat(succs, dim=0).long()

        return {
            "rules": rules[0],
            "state": init_state[0],
            "labels" : succs
        }


