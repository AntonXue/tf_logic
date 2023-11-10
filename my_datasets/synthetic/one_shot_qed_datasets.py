from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

import logic.prop_logic as proplog


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


class OneShotQedEmbedsDataset(Dataset):
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

class OneShotQedTextDataset(Dataset):
    pass



