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
        dataset_len: int,
        chain_len: int = 3,
        seed: int = 1234
    ):
        self.num_rules = num_rules
        self.num_vars = num_vars
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.chain_len = chain_len
        self.dataset_len = dataset_len
        self.seed = seed

    def __len__(self):
        """ We can indefinitely generate more, but set an artificial cap """
        return self.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)  # How to guarantee determinism
        rules_dict = logic.random_rules_with_chain(
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            chain_len = self.chain_len,
            return_dict = True)
        rules, bad_bits = rules_dict["rules"], rules_dict["bad_bits"]

        proof = logic.prove_theorem(rules.unsqueeze(0), torch.ones(1,self.num_vars))
        thm = proof["states"][0,-1] # The theorem is the iteration fixpoint
        label = torch.tensor(1).long()

        # Flip a coin to attempt to make it unprovable, if possible
        if torch.randn(()) > 0 and thm[bad_bits[0]] == 0:
                thm[bad_bits[0]] = 1
                label = torch.tensor(0).long()

        return {
            "rules" : rules,
            "theorem" : thm,
            "labels" : label
        }


class OneShotStringDataset(Dataset):
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
        seed: int = 1234,
        tokenizer: object = None,
        padding: str = "longest"
    ):
        self.num_rules = num_rules
        self.num_vars = num_vars
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.theorem_prob = theorem_prob
        self.dataset_len = dataset_len
        self.seed = seed
        self.tokenizer = tokenizer
        self.padding = padding

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
            conseq_prob = self.conseq_prob)

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
        
        encoding = self.tokenizer(entry_str, truncation=True, padding=self.padding)
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
        chain_len: int = 3,
        seed: int = 1234
    ):
        self.num_rules = num_rules
        self.num_vars = num_vars
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.state_prob = state_prob
        self.chain_len = chain_len
        self.dataset_len = dataset_len
        self.seed = seed

    def __len__(self):
        """ We can indefinitely generate more, but set an artificial cap """
        return self.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)
        rules = logic.random_rules_with_chain(
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            chain_len = self.chain_len)

        state = (torch.rand(1, self.num_vars) < self.state_prob).long()
        succ, _ = logic.step_rules(rules.unsqueeze(0), state)

        return {
            "rules" : rules,
            "state" : state[0],
            "labels" : succ[0]
        }


class NextStateStringDataset(Dataset):
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
        seed: int = 1234
    ):
        self.num_rules = num_rules
        self.num_vars = num_vars
        self.num_steps = num_steps
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.state_prob = state_prob
        self.dataset_len = dataset_len
        self.seed = seed

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)
        rules = logic.random_rules(
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            chain_len = self.num_steps)

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


class AutoRegKStepsStringDataset(Dataset):
    pass

