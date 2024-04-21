import math
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
        do_padding: bool = True,
    ):
        assert num_vars > 2
        assert num_rules_range[0] > 2 and num_rules_range[0] <= num_rules_range[1]
        assert ante_prob_range[0] > 0.0 and ante_prob_range[1] < 1.0
        assert ante_prob_range[0] <= ante_prob_range[1]
        assert conseq_prob_range[0] > 0.0 and conseq_prob_range[1] < 1.0
        assert conseq_prob_range[0] <= conseq_prob_range[1]

        self.num_vars = num_vars
        self.num_rules_range = num_rules_range
        self.ante_prob_range = ante_prob_range
        self.conseq_prob_range = conseq_prob_range
        self.chain_len_range = chain_len_range
        self.dataset_len = dataset_len
        self.do_padding = do_padding
        self.max_seq_len = num_rules_range[1] + 1

    def __len__(self):
        return self.dataset_len

    def get_random_rules(self):
        num_rules = torch.randint(self.num_rules_range[0], self.num_rules_range[1]+1, ())
        chain_len = torch.randint(self.chain_len_range[0], self.chain_len_range[1]+1, ())
        min_ap, max_ap = self.ante_prob_range
        min_bp, max_bp = self.conseq_prob_range
        ap = (max_ap - min_ap) * torch.rand(()) + min_ap
        bp = (max_bp - min_bp) * torch.rand(()) + min_bp

        return random_rules_with_chain(
            num_vars = self.num_vars,
            num_rules = num_rules,
            ante_prob = ap,
            conseq_prob = bp,
            chain_len = chain_len,
            return_dict = True,
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


class AutoregKStepsTokensDataset(Dataset):
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



class DiamondRuleTokensDataset(Dataset):
    """
    Inject a structure which has rules of form:
        
    Rules:
        a -> b
        a -> c
        b,c -> d
        (... others ...)

    This can then be done in k=3 autoregressive steps starting from the state (a)
    """
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        dataset_len: int,
        hot_prob: Optional[float] = None,
        do_padding: bool = True
    ):
        assert num_vars > 4
        assert num_rules_range[0] <= num_rules_range[1]
        assert num_rules_range[0] > 5

        self.num_vars = num_vars
        self.num_rules_range = num_rules_range
        self.dataset_len = dataset_len
        self.hot_prob = (1.0 / num_vars) if hot_prob is None else hot_prob
        self.do_padding = do_padding

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        num_rules = torch.randint(self.num_rules_range[0], self.num_rules_range[1]+1, ())

        # The a,b,c,d bits for the rules. The e bit is used to blacklist rules.
        abcde = torch.randperm(self.num_vars)[:5]
        a, b, c, d, e = abcde

        diamond_antes = torch.zeros(3, self.num_vars)
        diamond_conseqs = torch.zeros(3, self.num_vars)

        # a -> b
        diamond_antes[0,a] = 1
        diamond_conseqs[0,b] = 1

        # a -> c
        diamond_antes[1,a] = 1
        diamond_conseqs[1,c] = 1

        # b, c -> d
        diamond_antes[2,b] = 1
        diamond_antes[2,c] = 1
        diamond_conseqs[2,d] = 1

        # Coalesce the diamond structure
        diamond_rules = torch.cat([diamond_antes, diamond_conseqs], dim=1)

        other_antes = torch.rand(num_rules-3, self.num_vars) < self.hot_prob
        other_antes[:,e] = 1
        other_conseqs = torch.rand(num_rules-3, self.num_vars) < self.hot_prob
        other_rules = torch.cat([other_antes, other_conseqs], dim=-1).long()
        rules = torch.cat([diamond_rules, other_rules], dim=0)

        if self.do_padding:
            pad_rules = torch.zeros(self.num_rules_range[1] - num_rules, 2 * self.num_vars)
            rules = torch.cat([rules, pad_rules], dim=0)

        rules = rules[torch.randperm(rules.size(0))]    # Shuffle

        init_state = torch.zeros(self.num_vars)
        init_state[a] = 1
        init_token = torch.cat([torch.zeros(self.num_vars), init_state])
        tokens = torch.cat([rules, init_token.view(1,-1)], dim=0)

        # We can predictably generate the subsequent states
        labels = torch.zeros(2, self.num_vars)

        # Step 1
        labels[0,a] = 1
        labels[0,b] = 1
        labels[0,c] = 1
        
        # Step 2
        labels[1,a] = 1
        labels[1,b] = 1
        labels[1,c] = 1
        labels[1,d] = 1

        return {
            "tokens": tokens.long(),
            "labels": labels.long(),
            "abcde": torch.stack([a,b,c,d,e]).long(),
        }


