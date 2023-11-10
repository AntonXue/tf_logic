from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import logic.prop_logic as proplog


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


class AutoRegFixedStepsEmbedsDataset(Dataset):
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


class AutoRegFixedStepsTextDataset(Dataset):
    pass


