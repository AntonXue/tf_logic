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



