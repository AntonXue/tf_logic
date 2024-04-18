import torch
from torch.utils.data import Dataset

from .utils.logic_utils import *
from my_datasets.task_datasets import AutoregKStepsTokensDataset


class BadSuffixDataset(Dataset):
    """ Dataset for performing attacks on models trained with AutoregKSteps """
    def __init__(
        self,
        reasoner_dataset: AutoregKStepsTokensDataset,
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
    def __init__(
        self,
        reasoner_dataset: AutoregKStepsTokensDataset,
        dataset_len: Optional[int] = None,
    ):
        self.reasoner_dataset = reasoner_dataset
        self.num_vars = reasoner_dataset.num_vars
        self.num_steps = reasoner_dataset.num_steps
        self.dataset_len = len(reasoner_dataset) if dataset_len is None else dataset_len
        assert self.dataset_len <= len(reasoner_dataset)

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

        # WLOG, suppress the first rule
        supp_ind = 0 if (oks.sum() == 0) else oks.nonzero().view(-1)[0]
        supp_rule = tokens[supp_ind]
        alt_rules = torch.cat([tokens[:supp_ind], tokens[supp_ind+1:]], dim=0)

        s, alt_succs = state, []
        for k in range(self.reasoner_dataset.num_steps):
            s , _ = step_rules(alt_rules.unsqueeze(0), s.unsqueeze(0))
            s = s[0]
            alt_succs.append(s)
        alt_succs = torch.stack(alt_succs)

        # Need to calculate the new label that's supposed to happen
        return {
            "tokens": tokens,
            "supp_rule": supp_rule,
            "labels": alt_succs,
        }


