import math
import torch
from torch.utils.data import Dataset

from .utils.logic_utils import *
from .utils.string_utils import *


def hot(i,n):
    return F.one_hot(i,n)


class AutoregFixedProbTokensDataset(Dataset):
    """
    Randomized dataset for autoregressive stepping
    [rules] [prev_states]
    """
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        ante_prob: float,
        conseq_prob: float,
        state_prob: float,
        chain_len_range: tuple[int, int],
        num_prevs_range: tuple[int, int],
        num_steps: int,
        dataset_len: int,
        do_padding: bool = True,
    ):
        assert num_vars > 2
        assert num_rules_range[0] > 2 and num_rules_range[0] <= num_rules_range[1]

        assert chain_len_range[0] <= chain_len_range[1]
        assert num_rules_range[0] > chain_len_range[1] + 2
        assert num_prevs_range[0] >= 1
        assert num_prevs_range[0] <= num_prevs_range[1]
        assert num_prevs_range[1] + num_steps <= num_vars

        self.num_vars = num_vars
        self.num_rules_range = num_rules_range
        self.ante_prob = ante_prob
        self.conseq_prob = conseq_prob
        self.state_prob = state_prob
        self.chain_len_range = chain_len_range
        self.num_prevs_range = num_prevs_range
        self.num_steps = num_steps
        self.dataset_len = dataset_len

        self.max_seq_len = num_rules_range[1] + num_prevs_range[1]
        self.do_padding = do_padding

    def __len__(self):
        return self.dataset_len

    def get_random_rules(self):
        num_rules = torch.randint(self.num_rules_range[0], self.num_rules_range[1]+1, ())
        chain_len = torch.randint(self.chain_len_range[0], self.chain_len_range[1]+1, ())
        return random_rules_with_chain(
            num_vars = self.num_vars,
            num_rules = num_rules,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            state_prob = self.state_prob,
            chain_len = chain_len,
            return_dict = True,
        )

    def __getitem__(self, idx):
        num_vars = self.num_vars
        rules_dict = self.get_random_rules()
        rules, states = rules_dict["rules"], rules_dict["states"]
        num_rules = rules.size(0)

        # Generate the previous states
        num_prevs = torch.randint(self.num_prevs_range[0], self.num_prevs_range[1]+1, ())
        prevs = states[:num_prevs]
        succs = states[num_prevs:num_prevs+self.num_steps]

        # Pad and shuffle the rules if necessary
        if self.do_padding:
            pad_len = self.max_seq_len - num_rules - num_prevs
            pad_rules = torch.zeros(pad_len, 2*num_vars)
            rules = torch.cat([rules, torch.zeros(pad_len, 2*num_vars)], dim=0)
            rules = rules[torch.randperm(rules.size(0))]

        # Prepare the output tokens
        prev_tokens = torch.cat([torch.zeros(num_prevs, num_vars), prevs], dim=1)
        all_tokens = torch.cat([rules, prev_tokens], dim=0)

        return {
            "tokens": all_tokens,
            "labels": succs
        }


class AutoregScaledProbTokensDataset(Dataset):
    """ [rules] [prev_states] """
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        exp_hots: float,
        chain_len_range: tuple[int, int],
        num_prevs_range: tuple[int, int],
        num_steps: int,
        dataset_len: int,
        do_padding: bool = True,
    ):
        assert num_vars > 2
        assert num_rules_range[0] > 2 and num_rules_range[0] <= num_rules_range[1]
        assert chain_len_range[0] <= chain_len_range[1]
        assert num_rules_range[0] > chain_len_range[1] + 2
        assert num_prevs_range[0] >= 1
        assert num_prevs_range[0] <= num_prevs_range[1]
        assert num_prevs_range[1] + num_steps <= num_vars

        self.num_vars = num_vars
        self.num_rules_range = num_rules_range
        self.exp_hots = exp_hots
        self.ante_prob = exp_hots / num_vars
        self.conseq_prob = exp_hots / num_vars
        self.state_prob = exp_hots / num_vars
        self.chain_len_range = chain_len_range
        self.num_prevs_range = num_prevs_range
        self.num_steps = num_steps
        self.dataset_len = dataset_len

        self.max_seq_len = num_rules_range[1] + num_prevs_range[1]
        self.do_padding = do_padding

    def __len__(self):
        return self.dataset_len

    def get_random_rules(self):
        num_rules = torch.randint(self.num_rules_range[0], self.num_rules_range[1]+1, ())
        chain_len = torch.randint(self.chain_len_range[0], self.chain_len_range[1]+1, ())
        return random_rules_with_chain(
            num_vars = self.num_vars,
            num_rules = num_rules,
            ante_prob = self.ante_prob,
            conseq_prob = self.conseq_prob,
            state_prob = self.state_prob,
            chain_len = chain_len,
            return_dict = True,
        )

    def __getitem__(self, idx):
        num_vars = self.num_vars
        rules_dict = self.get_random_rules()
        rules, states = rules_dict["rules"], rules_dict["states"]
        num_rules = rules.size(0)

        # Generate the previous states
        num_prevs = torch.randint(self.num_prevs_range[0], self.num_prevs_range[1]+1, ())
        prevs = states[:num_prevs]
        succs = states[num_prevs:num_prevs+self.num_steps]

        # Pad and shuffle the rules if necessary
        if self.do_padding:
            pad_len = self.max_seq_len - num_rules - num_prevs
            pad_rules = torch.zeros(pad_len, 2*num_vars)
            rules = torch.cat([rules, torch.zeros(pad_len, 2*num_vars)], dim=0)
            rules = rules[torch.randperm(rules.size(0))]

        # Prepare the output tokens
        prev_tokens = torch.cat([torch.zeros(num_prevs, num_vars), prevs], dim=1)
        all_tokens = torch.cat([rules, prev_tokens], dim=0)

        return {
            "tokens": all_tokens,
            "labels": succs
        }


class AutoregDiamondTokensDataset(Dataset):
    """
        {a} -> {abc} -> {abcd} -> {abcdef}
    """
    def __init__(
        self,
        num_vars: int,
        num_rules: int,
        exp_hots: float,
        dataset_len: int
    ):
        assert num_vars > 7
        self.num_vars = num_vars
        self.num_rules = num_rules
        self.exp_hots = exp_hots
        self.hot_prob = exp_hots / num_vars
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n, r, p = self.num_vars, self.num_rules, self.hot_prob
        abcdefg = torch.randperm(n)[:7]
        a, b, c, d, e, f, g = abcdefg

        special_rules = torch.stack([
            torch.cat([hot(a,n), hot(b,n)]), # a -> b
            torch.cat([hot(a,n), hot(c,n)]), # a -> c
            torch.cat([hot(b,n) + hot(c,n), hot(d,n)]), # b,c -> d
            torch.cat([hot(d,n), hot(e,n)]), # d -> e
            torch.cat([hot(d,n), hot(f,n)]), # d -> f
        ]).long()

        num_others = r - special_rules.size(0)
        other_antes = (torch.rand(num_others, n) < p).long()
        other_antes[:,g] = 1
        other_conseqs = (torch.rand(num_others, n) < p).long()
        other_rules = torch.cat([other_antes, other_conseqs], dim=-1)
        rules = torch.cat([special_rules, other_rules], dim=0)
        rules = rules[torch.randperm(rules.size(0))]

        init_token = torch.cat([torch.zeros(n), hot(a,n)])
        tokens = torch.cat([rules, init_token.view(1,2*n)], dim=0)

        labels = torch.stack([
            hot(a,n) + hot(b,n) + hot(c,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n),
        ]).long()

        return {
            "tokens": tokens,
            "labels": labels,
            "abcdefg": abcdefg
        }


class AutoregCustomTokensDataset(Dataset):
    """
        a -> b
        a -> c
        a -> d
        b,c -> e
        c,d -> f
        e,f -> g

        Proof states: {a} -> {abcd} -> {abcdef} -> {abcdefg}
    """
    def __init__(
        self,
        num_vars: int,
        num_rules: int,
        exp_hots: float,
        dataset_len: int
    ):
        assert num_vars >= 8
        self.num_vars = num_vars
        self.num_rules = num_rules
        self.exp_hots = exp_hots
        self.hot_prob = exp_hots / num_vars
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n, r, p = self.num_vars, self.num_rules, self.hot_prob
        abcdefgh = torch.randperm(n)[:8]
        a, b, c, d, e, f, g, h = abcdefgh

        special_rules = torch.stack([
            torch.cat([hot(a,n), hot(b,n)]), # a -> b
            torch.cat([hot(a,n), hot(c,n)]), # a -> c
            torch.cat([hot(a,n), hot(d,n)]), # a -> d
            torch.cat([hot(b,n) + hot(c,n), hot(e,n)]), # b,c -> e
            torch.cat([hot(c,n) + hot(d,n), hot(f,n)]), # c,d -> f
            torch.cat([hot(e,n) + hot(f,n), hot(g,n)]), # e,f -> g
        ]).long()

        num_others = r - special_rules.size(0)
        other_antes = (torch.rand(num_others, n) < p).long()
        other_antes[:,h] = 1
        other_conseqs = (torch.rand(num_others, n) < p).long()
        other_rules = torch.cat([other_antes, other_conseqs], dim=-1)
        rules = torch.cat([special_rules, other_rules], dim=0)
        rules = rules[torch.randperm(rules.size(0))]

        init_token = torch.cat([torch.zeros(n), hot(a,n)])
        tokens = torch.cat([rules, init_token.view(1,2*n)], dim=0)

        labels = torch.stack([
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n),
            hot(a,n) + hot(b,n) + hot(c,n) + hot(d,n) + hot(e,n) + hot(f,n) + hot(g,n)
        ]).long()

        return {
            "tokens": tokens,
            "labels": labels,
            "abcdefgh": abcdefgh
        }


