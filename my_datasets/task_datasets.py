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
        num_fillers: int = 2
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
        self.num_fillers = num_fillers

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
            num_rules = num_rules,
            num_vars = self.num_vars,
            ante_prob = ap,
            conseq_prob = bp,
            chain_len = chain_len,
            return_dict = True,
            num_fillers = self.num_fillers
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
        self.inner_dataset = OneShotTokensDataset(
            num_vars = num_vars,
            num_rules_range = num_rules_range,
            ante_prob_range = ante_prob_range,
            conseq_prob_range = conseq_prob_range,
            chain_len_range = chain_len_range,
            dataset_len = dataset_len,
            num_fillers = num_fillers
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


class TiledAutoregKStepsTokensDataset(Dataset):
    """ [zero_pads] [rules] [prestepped_states]
    """
    def __init__(
        self,
        num_vars: int,
        num_rules_range: tuple[int, int],
        ante_prob_range: tuple[float, float],
        conseq_prob_range: tuple[float, float],
        chain_len_range: tuple[int, int],
        num_presteps_range: tuple[int, int],
        num_todo_steps: int,
        dataset_len: int,
        do_padding: bool = True,
    ):
        assert num_vars > 2
        assert num_rules_range[0] > 2 and num_rules_range[0] <= num_rules_range[1]
        assert ante_prob_range[0] > 0.0 and ante_prob_range[1] < 1.0
        assert ante_prob_range[0] <= ante_prob_range[1]
        assert conseq_prob_range[0] > 0.0 and conseq_prob_range[1] < 1.0
        assert conseq_prob_range[0] <= conseq_prob_range[1]
        assert chain_len_range[0] <= chain_len_range[1]
        assert num_rules_range[0] > chain_len_range[1] + 2

        self.num_vars = num_vars
        self.num_rules_range = num_rules_range
        self.ante_prob_range = ante_prob_range
        self.conseq_prob_range = conseq_prob_range
        self.chain_len_range = chain_len_range
        self.num_presteps_range = num_presteps_range
        self.num_todo_steps = num_todo_steps
        self.dataset_len = dataset_len
        # max_num_presteps + 1 because we begin pre-stepping at the zero state
        self.max_seq_len = num_rules_range[1] + num_presteps_range[1] + 1
        self.do_padding = do_padding

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
            num_rules = num_rules,
            num_vars = self.num_vars,
            ante_prob = ap,
            conseq_prob = bp,
            chain_len = chain_len,
            return_dict = True,
            num_fillers = 0
        )

    def __getitem__(self, idx):
        num_vars = self.num_vars
        rules_dict = self.get_random_rules()
        rules = rules_dict["rules"]
        num_rules = rules.size(0)

        num_presteps = torch.randint(self.num_presteps_range[0], self.num_presteps_range[1]+1, ())
        state = torch.zeros(1, num_vars)
        prestep_states = (state,)
        for _ in range(num_presteps):
            state, _ = step_rules(rules[None,...], state)
            prestep_states += (state,)

        todo_states = ()
        for _ in range(self.num_todo_steps):
            state, _ = step_rules(rules[None,...], state)
            todo_states += (state,)

        if self.do_padding:
            pad_len = self.max_seq_len - num_rules - len(prestep_states)

        all_tokens = torch.cat([
            torch.zeros(pad_len, 2*num_vars), # zero paddings
            rules,                      # the rules
            torch.cat([                 # prestepped states
                torch.zeros(len(prestep_states), num_vars),
                torch.cat(prestep_states, dim=0)
            ], dim=1)
        ])

        return {
            "tokens": all_tokens,
            "labels": torch.cat(todo_states, dim=0)
        }


