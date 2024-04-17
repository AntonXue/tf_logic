import torch
from torch.utils.data import Dataset

from .utils.logic_utils import *
from my_datasets.task_datasets import AutoregKStepsTokensDataset


class AttackWrapperDataset(Dataset):
    """ Dataset for performing attacks on models trained with AutoregKSteps """
    def __init__(
        self,
        reasoner_dataset: AutoregKStepsTokensDataset,
        num_attack_tokens: int,
        dataset_len: int,
    ):
        assert dataset_len <= len(reasoner_dataset)
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.num_attack_tokens = num_attack_tokens
        self.dataset_len = dataset_len

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
    def __init__(
        self,
        reasoner_dataset: AutoregKStepsTokensDataset,
        dataset_len: int,
    ):
        assert dataset_len <= len(reasoner_dataset)
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        item = self.reasoner_dataset[idx]
        tokens = item["tokens"]

        state = tokens[-1,self.num_vars:]
        antes, conseqs = tokens.chunk(2, dim=1)

        # Find what to ignore
        ante_oks = (antes <= state.view(1,-1)).sum(dim=1) == self.num_vars
        conseq_oks = ((conseqs - state.view(1,-1)) > 0).sum(dim=1) > 0
        oks = (ante_oks * conseq_oks).long() # (L,), the length of the token sequence

        # Find the first acceptable index to delete. Otherwise pick the first one
        ign_ind = 0 if (oks.sum() == 0) else oks.nonzero().view(-1)[0]
        ign_rule = tokens[ign_ind]

        # Need to calculate the new label that's supposed to happen
        spliced_tokens = torch.cat([tokens[:ign_ind], tokens[ign_ind+1:]], dim=0)
        alt_succ, _ = step_rules(spliced_tokens.unsqueeze(0), state.unsqueeze(0))
        orig_succ, _ = step_rules(tokens.unsqueeze(0), state.unsqueeze(0))

        return {
            "tokens": tokens,
            "state": state,
            "original_succ": orig_succ[0],
            "ignored_ind": ign_ind,
            "ignored_rule": ign_rule,
            "labels": alt_succ[0],
        }


