import math
import torch
from torch.utils.data import Dataset
from typing import *
from .task_datasets import *



class SmallTfSuccTokensDataset(Dataset):
    """ [rules] [prev_states] """
    def __init__(
        self,
        num_vars: int,
        dataset_len: int,
        num_rules: Optional[int] = None,
        ante_prob: float = 0.5,
        conseq_prob: float = 0.5
    ):
        self.num_vars = num_vars
        self.num_rules = (1 * num_vars) if num_rules is None else num_rules
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.dataset_len = dataset_len

    @property
    def desc_str(self):
        return f"Dsimple_n{self.num_vars}_ap{self.ante_prob}_bp{self.conseq_prob}"

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n, r = self.num_vars, self.num_rules
        all_as = (torch.rand(r, n) < self.ante_prob).long()
        all_bs = (torch.rand(r, n) < self.conseq_prob).long()

        # Determine which antes to use in the current state
        todo = (torch.rand(r, 1) < 1/r).long()
        state = (all_as * todo).sum(dim=0).clamp(0,1)

        # Tokens is just the rules with the state concated on
        rules_tokens = torch.cat([all_as, all_bs], dim=1)
        state_token = torch.cat([torch.zeros(n), state])
        all_tokens = torch.cat([rules_tokens, state_token.view(1,2*n)], dim=0)

        succ, _ = step_rules(all_tokens.unsqueeze(0), state.unsqueeze(0))
        succ = succ.squeeze()

        return {
            "tokens": all_tokens,
            "labels": succ
        }


class SmallAutoreg1StepTokensDataset(Dataset):
    def __init__(
        self,
        num_vars: int,
        dataset_len: int,
        ante_prob: float = 0.5,
        conseq_prob: float = 0.5
    ):
        self.num_vars = num_vars
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob

        self.inner_dataset = AutoregKStepsTokensDataset(
            num_vars = num_vars,
            num_rules_range = (num_vars, 2*num_vars),
            ante_prob_range = (ante_prob, ante_prob),
            conseq_prob_range = (conseq_prob, conseq_prob),
            chain_len_range = (2, num_vars//2),
            num_prevs_range = (1, num_vars//2),
            num_steps = 1,
            dataset_len = dataset_len
        )

    @property
    def desc_str(self):
        return f"Dar1s_n{self.num_vars}_ap{self.ante_prob}_bp{self.conseq_prob}"

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, idx):
        res = self.inner_dataset[idx]
        return {
            "tokens": res["tokens"],
            "labels": res["labels"][0].view(self.num_vars)
        }


