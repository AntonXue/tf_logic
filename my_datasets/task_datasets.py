import torch
from torch.utils.data import Dataset

from .utils.logic_utils import *
from .utils.string_utils import *

class OneShotTokensDataset(Dataset):
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        ante_prob_range: tuple[float, float],
        conseq_prob_range: tuple[float, float],
        chain_len_range: tuple[int, int],
        dataset_len: int,
        do_padding: bool = True
    ):
        assert num_vars > 2
        assert num_rules_range[0] > 2 and num_rules_range[0] <= num_rules_range[1]
        assert ante_prob_range[0] > 0.0 and ante_prob_range[1] < 1.0
        assert ante_prob_range[0] <= ante_prob_range[1]
        assert conseq_prob_range[0] > 0.0 and conseq_prob_range[1] < 1.0
        assert conseq_prob_range[0] <= conseq_prob_range[1]

        self.num_vars = num_vars
        self.num_rules_range = num_rules_range
        self.ap_range = ante_prob_range
        self.bp_range = conseq_prob_range
        self.chain_len_range = chain_len_range
        self.dataset_len = dataset_len
        self.do_padding = do_padding
        self.max_seq_len = num_rules_range[1] + 1

    def __len__(self):
        return self.dataset_len

    def get_random_rules(self):
        num_rules = torch.randint(self.num_rules_range[0], self.num_rules_range[1]+1, ())
        chain_len = torch.randint(self.chain_len_range[0], self.chain_len_range[1]+1, ())
        _ap, _bp = torch.rand(2)
        ap = (self.ap_range[1] - self.ap_range[0]) * _ap + self.ap_range[0]
        bp = (self.bp_range[1] - self.bp_range[0]) * _bp + self.bp_range[0]

        return random_rules_with_chain(
            num_rules = num_rules,
            num_vars = self.num_vars,
            ante_prob = ap,
            conseq_prob = bp,
            chain_len = chain_len,
            return_dict = True
        )

    def __getitem__(self, idx):
        # Random numbers
        num_vars = self.num_vars
        rules_dict = self.get_random_rules()
        rules, bad_bit = rules_dict["rules"], rules_dict["bad_bit"]
        num_rules = rules.size(0)

        if self.do_padding:
            pad_len = self.max_seq_len - num_rules - 1
            pad_rules = torch.zeros(pad_len, 2*num_vars)
            rules = torch.cat([rules, pad_rules], dim=0).long()
            num_rules = rules.size(0)

        proof = prove_theorem(rules[None,...], torch.ones(1, num_vars))
        thm = proof["states"][0,-1] # The theorem is the iteration fixpoint
        labels = torch.tensor(1).long()

        if torch.rand(()) > 0.5:
            thm[bad_bit] = 1
            labels = torch.tensor(0).long()

        all_tokens = torch.cat([
                torch.cat([torch.zeros(num_rules,1), rules], dim=1),
                torch.cat([torch.ones(1), torch.zeros(num_vars), thm])[None,...]
            ], dim=0)

        all_tokens = all_tokens[torch.randperm(all_tokens.size(0))]

        return {
            "tokens": all_tokens,
            "labels": labels
        }


class OneShotStringDataset(Dataset):
    """ For task of checking one-shot QED, generate a bunch of random rules 
    The rules and theorem are represented as strings"""
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        ante_prob_range: tuple[float, float],
        conseq_prob_range: tuple[float, float],
        chain_len_range: tuple[int, int],
        dataset_len: int,
        tokenizer: object = None,
        padding: str = "longest"
    ):
        self.tokenizer = tokenizer
        self.padding = padding

        self.inner_dataset = OneShotTokensDataset(
            num_vars = num_vars,
            num_rules_range = num_rules_range,
            ante_prob_range = ante_prob_range,
            conseq_prob_range = conseq_prob_range,
            chain_len_range = chain_len_range,
            dataset_len = dataset_len,
        )

    def __len__(self):
        """ We can indefinitely generate more, but set an artificial cap """
        return len(self.inner_dataset)

    def __getitem__(self, idx):
        num_vars = self.inner_dataset.num_vars
        rules_dict = self.inner_dataset.get_random_rules()
        rules, bad_bit = rules_dict["rules"], rules_dict["bad_bit"]

        proof = prove_theorem(rules[None,...], torch.ones(1,num_vars))
        thm = proof["states"][0,-1]
        labels = torch.tensor(1).long()

        if torch.rand(()) > 0.5:
            thm[bad_bit] = 1
            labels = torch.tensor(0).long()

        entry = {
            "rules": rules,
            "theorem": thm,
            "labels": labels
        }

        entry_str = get_string_rep(entry)
        if not self.tokenizer:
            return Exception("Tokenizer not provided.")

        encoding = self.tokenizer(entry_str, truncation=True, padding=self.padding)
        return {
            "data": entry_str,
            "labels": labels,
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask
        }


class NextStateTokensDataset(Dataset):
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        num_states_range: tuple[int, int],
        ante_prob_range: tuple[float, float],
        conseq_prob_range: tuple[float, float],
        state_prob_range: tuple[float, float],
        dataset_len: int,
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
        self.do_padding = do_padding
        self.max_seq_len = num_rules_range[1] + num_states_range[1]

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # Random numbers
        num_vars = self.num_vars
        num_states = torch.randint(self.num_states_range[0], self.num_states_range[1]+1, ())
        num_rules = torch.randint(self.num_rules_range[0], self.num_rules_range[1]+1, ())
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
        next_state, _ = step_rules(all_rules[None,...], big_state)

        # tokens
        state_tokens = torch.cat([torch.ones(num_states,1), 0*all_states, all_states], dim=1)
        rule_tokens = torch.cat([torch.zeros(num_rules,1), all_rules], dim=1)
        all_tokens = torch.cat([rule_tokens, state_tokens], dim=0)
        all_tokens = all_tokens[torch.randperm(all_tokens.size(0))]

        return {
            "tokens": all_tokens,
            "labels": next_state[0],
        }


class AutoregKStepsTokensDataset(Dataset):
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        ante_prob_range: tuple[float, float],
        conseq_prob_range: tuple[float, float],
        chain_len_range: tuple[int, int],
        num_steps: int,
        dataset_len: int,
        do_padding: bool = True
    ):
        self.inner_dataset = OneShotTokensDataset(
            num_vars = num_vars,
            num_rules_range = num_rules_range,
            ante_prob_range = ante_prob_range,
            conseq_prob_range = conseq_prob_range,
            chain_len_range = chain_len_range,
            dataset_len = dataset_len,
        )

        self.num_steps = num_steps
        self.do_padding = do_padding
        self.max_seq_len = num_rules_range[1]

    def __len__(self):
        """ We can indefinitely generate more, but set an artificial cap """
        return len(self.inner_dataset)       

    def __getitem__(self, idx):
        num_vars = self.inner_dataset.num_vars
        rules_dict = self.inner_dataset.get_random_rules()
        rules = rules_dict["rules"]
        num_rules = rules.size(0)

        if self.do_padding:
            pad_len = self.max_seq_len - num_rules
            pad_rules = torch.zeros(pad_len, 2*num_vars)
            rules = torch.cat([rules, pad_rules], dim=0).long()
            num_rules = rules.size(0)

        state = torch.zeros(1, num_vars).long()
        succs = ()
        for t in range(self.num_steps):
            state, _ = step_rules(rules[None,...], state)
            succs += (state,)

        succs = torch.cat(succs, dim=0).long()
        all_tokens = torch.cat([torch.zeros(num_rules,1), rules], dim=1)
        all_tokens = all_tokens[torch.randperm(all_tokens.size(0))]
        
        return {
            "tokens": all_tokens,
            "labels": succs
        }


