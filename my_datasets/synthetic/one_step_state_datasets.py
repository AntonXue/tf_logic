from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import logic.prop_logic as proplog


@dataclass
class OneStepStateDatasetConfig:
    num_rules: int
    num_vars: int
    ante_prob: float
    conseq_prob: float
    state_prob: float
    dataset_len: int
    ensure_facts: bool = True
    seed: int = 1234


class OneStepStateEmbedsDataset(Dataset):
    """ For task of generating the next state, generate a bunch of random rules """
    def __init__(self, config: OneStepStateDatasetConfig):
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


class OneStepStateTextDataset(Dataset):
	pass


