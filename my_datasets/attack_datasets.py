import torch
from torch.utils.data import Dataset

from .utils.logic_utils import *
from my_datasets.task_datasets import AutoregKStepsTokensDataset


class ForceOutputWithAppendedAttackTokensDataset(Dataset):
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

        # Tokens
        state_tokens = torch.cat([torch.ones(num_states,1), 0*all_states, all_states], dim=1)
        rule_tokens = torch.cat([torch.zeros(num_rules,1), all_rules], dim=1)
        all_tokens = torch.cat([rule_tokens, state_tokens], dim=0)
        all_tokens = all_tokens[torch.randperm(all_tokens.size(0))]

        # Figure out the attack; big_state already has batch_size == 1
        succ_state, _ = step_rules(all_rules[None,...], big_state)
        succ_state = succ_state[0]
        target = (torch.randint(0,3,(1, num_vars)) - (torch.rand(num_vars) < sp) * succ_state).clamp(0,1)

        return {
            "tokens": all_tokens,
            "target": target
        }
    
class AutoregKStepsTokensAttackDataset(Dataset):
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        ante_prob_range: tuple[float, float],
        conseq_prob_range: tuple[float, float],
        chain_len_range: tuple[int, int],
        num_steps: int,
        dataset_len: int,
        do_padding: bool = True,
        num_fillers: int = 2
    ):
        self.unperturbed_dataset = AutoregKStepsTokensDataset(
            num_vars=num_vars,
            num_rules_range=num_rules_range,
            ante_prob_range=ante_prob_range,
            conseq_prob_range=conseq_prob_range,
            chain_len_range=chain_len_range,
            num_steps=num_steps,
            dataset_len=dataset_len,
            do_padding=do_padding,
            num_prevs_range=(1, chain_len_range[0])
            # num_fillers=num_fillers
        )
        self.num_vars = num_vars

    def __len__(self):
        return len(self.unperturbed_dataset)
    
    def __getitem__(self, idx):
        unperturbed_item = self.unperturbed_dataset[idx]
        tokens = unperturbed_item["tokens"]
        # adv_target = shuffled target
        adv_target = torch.randint(0, 2, (1, self.num_vars))
        return {
            "tokens": tokens,
            "labels": adv_target
        }


