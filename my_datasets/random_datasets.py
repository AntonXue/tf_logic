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
            "rules": rules,
            "theorem": thm,
            "labels": label
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
                "rules": rules[0],
                "theorem": thm[0],
                "labels": qed[0]
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
            "rules": rules,
            "state": state[0],
            "labels": succ[0]
        }


class NextStateStringDataset(Dataset):
    pass


class NextStateFromTokensEmbedsDataset(Dataset):
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        num_states_range: tuple[int, int],
        ante_prob_range: tuple[float, float],
        conseq_prob_range: tuple[float, float],
        state_prob_range: tuple[float, float],
        dataset_len: int,
        seed: int = 1234,
        do_padding: bool = True
    ):
        assert num_vars > 2
        assert num_rules_range[0] > 2 and num_rules_range[0] <= num_rules_range[1]
        assert num_states_range[0] > 2 and num_states_range[0] <= num_states_range[1]
        assert ante_prob_range[0] > 0.0 and ante_prob_range[1] < 1.0
        assert ante_prob_range[0] <= ante_prob_range[1]
        assert conseq_prob_range[0] > 0.0 and conseq_prob_range[1] < 1.0
        assert conseq_prob_range[0] <= conseq_prob_range[1]
        assert state_prob_range[0] > 0.0 and state_prob_range[1] < 1.0
        assert state_prob_range[0] <= state_prob_range[1]

        self.num_vars = num_vars
        self.num_rules_range = num_rules_range
        self.num_states_range = num_states_range
        self.ap_range = ante_prob_range
        self.bp_range = conseq_prob_range
        self.sp_range = state_prob_range
        self.dataset_len = dataset_len
        self.seed = seed
        self.do_padding = do_padding
        self.max_seq_len = num_rules_range[1] + num_states_range[1]

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)

        # Random numbers
        num_vars = self.num_vars
        num_states = torch.randint(self.num_states_range[0], self.num_states_range[1]+1, (1,)).item()
        num_rules = torch.randint(self.num_rules_range[0], self.num_rules_range[1]+1, (1,)).item()
        _ap, _bp, _sp = torch.rand(3)
        ap = (self.ap_range[1] - self.ap_range[0]) * _ap + self.ap_range[0]
        bp = (self.bp_range[1] - self.bp_range[0]) * _bp + self.bp_range[0]
        sp = (self.sp_range[1] - self.sp_range[0]) * _sp + self.sp_range[0]

        # Generate some states, starting from a big_state from which we derive other stuff
        big_state = (torch.rand(1, num_vars) < sp).long()
        other_states = (torch.rand(num_states-1, num_vars) < 0.8) * big_state
        all_states = torch.cat([big_state, other_states], dim=0)

        # Generate some rules, in particular a special rule that is guaranteed to fire
        special_ante = (torch.rand(1, num_vars) < 0.8) * big_state
        special_conseq = (torch.rand(1, num_vars) < min(2*bp, 0.8)) * (1 - big_state)
        special_rule = torch.cat([special_ante, special_conseq], dim=1).long()

        other_antes = torch.rand(num_rules-1, num_vars) < ap
        other_conseqs = torch.rand(num_rules-1, num_vars) < bp
        other_rules = torch.cat([other_antes, other_conseqs], dim=1).long()
        all_rules = torch.cat([special_rule, other_rules], dim=0)

        if self.do_padding:
            pad_len = self.max_seq_len - num_rules - num_states
            pad_rules = torch.zeros(pad_len, 2*num_vars)
            all_rules = torch.cat([all_rules, pad_rules], dim=0).long()
            num_rules = all_rules.size(0)

        # big_state already has batch_size == 1
        next_state, _ = logic.step_rules(all_rules[None,...], big_state)

        # tokens
        state_tokens = torch.cat([torch.ones(num_states,1), 0*all_states, all_states], dim=1)
        rule_tokens = torch.cat([torch.zeros(num_rules,1), all_rules], dim=1)
        all_tokens = torch.cat([rule_tokens, state_tokens], dim=0)
        all_tokens = all_tokens[torch.randperm(all_tokens.size(0))]

        return {
            "tokens": all_tokens,
            "labels": next_state[0],
        }


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
        chain_len: int = 3,
        seed: int = 1234
    ):
        self.num_rules = num_rules
        self.num_vars = num_vars
        self.num_steps = num_steps
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.state_prob = state_prob
        self.chain_len = chain_len
        self.dataset_len = dataset_len
        self.seed = seed

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        torch.manual_seed(self.seed + idx)
        rules = logic.random_rules_with_chain(
            num_rules = self.num_rules,
            num_vars = self.num_vars,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            chain_len = self.chain_len,
            return_dict = False)

        init_state = (torch.rand(1, self.num_vars) < self.state_prob).long()
        tmp = init_state
        succs = ()
        for t in range(self.num_steps):
            tmp, _ = logic.step_rules(rules.unsqueeze(0), tmp)
            succs = succs + (tmp,)

        succs = torch.cat(succs, dim=0).long()

        return {
            "rules": rules,
            "state": init_state[0],
            "labels": succs
        }


class AutoRegKStepsStringDataset(Dataset):
    pass


