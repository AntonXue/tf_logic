from typing import Optional
import torch
import torch.nn.functional as F


""" Propositional Horn Clauses """

def random_rules_with_chain(
    num_vars: int,
    num_rules: int,
    ante_prob: float,
    conseq_prob: float,
    state_prob: float,
    chain_len: int,
    return_dict: bool = False
):
    """ Make one rule at a time (batch_size == 1) """
    assert num_rules > chain_len
    assert num_vars > chain_len
    r, n, ap, bp, sp = num_rules, num_vars, ante_prob, conseq_prob, state_prob

    order = torch.randperm(n)
    chain_bits = order[:chain_len]    # We need chain_len - 1 dedicated bits
    other_bits = order[chain_len:n-1] # The other stuffs
    bad_bit = order[-1] # The bad bit we wanna be excluding

    other_vec = torch.zeros(n).long()
    other_vec[other_bits] = 1

    init_state = (torch.rand(n) < sp).long()
    init_state[chain_bits] = 0
    init_state[bad_bit] = 0
    init_state = init_state

    # chain_bits, init_state, and other_vec are disjoint
    # a1, a2, a3, ..., ak, X
    #    /   /   /        /
    # b1, b2, b3, ..., bk
    chain_as = (torch.rand(chain_len, n) < ap) * init_state
    chain_bs = (torch.rand(chain_len, n) < bp) * other_vec
    for i, cb in enumerate(chain_bits):
        chain_bs[i, cb] = 1
        # The initial chain_a receives no chain_bit
        if i < chain_len - 1:
            chain_as[i+1, cb] = 1

    # Track all the states that could possibly be generated:
    #   all_states = [s0, s1, s2, ... sn]
    all_states = torch.zeros(num_vars + 1, num_vars)
    all_states[0] = init_state
    s = init_state
    for i, b in enumerate(chain_bs):
        s += b
        all_states[i+1] = s
    all_states[chain_len+1:] = s
    all_states = all_states.clamp(0,1).long()

    # Bad rules
    num_bad_rules = r - len(chain_as)
    bad_as = (torch.rand(num_bad_rules, n) < ap)
    bad_bs = (torch.rand(num_bad_rules, n) < bp)
    bad_as[:,bad_bit] = 1

    all_as = torch.cat([chain_as, bad_as], dim=0)
    all_bs = torch.cat([chain_bs, bad_bs], dim=0)
    all_rules = torch.cat([all_as, all_bs], dim=1).long()
    all_rules = all_rules[torch.randperm(r)]

    if return_dict:
        return {
            "rules": all_rules,
            "states": all_states,
            "chain_bits": chain_bits,
            "other_bits": other_bits,
            "bad_bit" : bad_bit,
        }
    else:
        return all_rules, all_states


""" Functionalities """


def split_abs(rules: torch.LongTensor):
    """ Split the antecedents and consequents """
    assert rules.ndim == 3 and rules.size(2) % 2 == 0
    return rules.chunk(2, dim=2)


def all_leq(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5):
    """ Do leq testing on the last coordinate with tolerance eps """
    assert x.ndim == y.ndim
    n = x.size(-1)
    z = (x <= y + eps).sum(dim=-1) # (*,)
    return (n <= z + eps).long()


def step_rules(rules: torch.LongTensor, state: torch.LongTensor):
    """ Apply one step of the rules. """
    N, r, n2 = rules.shape
    _, n = state.shape
    assert n2 == n*2
    a, b = split_abs(rules)         # (N,r,n), (N,r,n)
    hits = all_leq(a, state.view(N,1,n)) # (N,r)

    succ = (b * hits.view(N,r,1)).sum(dim=1)    # (N,n)
    succ = (state + succ).clamp(0,1)            # (N,n)
    return succ.long(), hits.long()   # (N,n), (N,r)


def kstep_rules(rules: torch.LongTensor, state: torch.LongTensor, num_steps: int):
    for t in range(num_steps):
        state, _ = step_rules(rules, state)
    return state


def prove_theorem(
    rules: torch.LongTensor,
    theorem: torch.LongTensor,
    init_state: Optional[torch.LongTensor] = None
):
    """ Run a proof and return a bunch of metadata """
    N, r, n2 = rules.shape
    _, n = theorem.shape
    assert n2 == n*2

    s = torch.zeros_like(theorem).long() if init_state is None else init_state
    all_states, all_hits, chain_len = [s], [], torch.zeros(N).long()
    for t in range(n):
        succ, h = step_rules(rules, s)
        all_states.append(succ)
        all_hits.append(h)
        chain_len += ((succ - s).sum(dim=1)) > 0    # Increment if got something new
        s = succ

    return {
        "rules": rules,
        "theorem": theorem,
        "qed": all_leq(theorem, s),
        "states": torch.stack(all_states, dim=1), # (N,n,n)
        "hits": torch.stack(all_hits, dim=1), # (N,n,r)
        "chain_len": chain_len, # (N,)
    }

