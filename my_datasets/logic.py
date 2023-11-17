from typing import Optional
import torch
import torch.nn.functional as F


""" Propositional Horn Clauses """

def random_rules_with_chain(
    num_rules: int,
    num_vars: int,
    ante_prob: float,
    conseq_prob: float,
    chain_len: int,
    num_fillers: int = 2,
    num_bad_bits: int = 2,
    return_dict: bool = False
):
    """ Make one rule at a time (batch_size == 1) """
    assert num_rules > num_fillers + chain_len + num_bad_bits
    r, n, pa, pb = num_rules, num_vars, ante_prob, conseq_prob

    # Figure out the chain stuff, and what bits to avoid in the conseq
    order = torch.arange(0, n)[torch.randperm(n)]
    good_bits, bad_bits = order[:chain_len], order[-num_bad_bits:]

    # The filler bits, which all have the first good bit in the conseq
    all_as, all_bs = [], []
    for _ in range(num_fillers):
        b = (torch.rand(n) < pb).long()
        b[bad_bits] = 0
        if len(good_bits) > 0:
            b[good_bits[0]] = 1
        all_as.append(torch.zeros(n).long())
        all_bs.append(b)

    all_facts = sum(all_bs).clamp(0,1)  # All the facts so far

    for k in range(len(good_bits) - 1):
        gk, gk1 = F.one_hot(good_bits[k], n), F.one_hot(good_bits[k+1], n)
        a = (gk + (all_facts * (torch.rand(n) < pa))).clamp(0,1)
        b = (gk1 + (all_facts * (torch.rand(n) < pa))).clamp(0,1)
        all_as.append(a)
        all_bs.append(b)
        all_facts = (all_facts + b).clamp(0,1)

    for _ in range(r - len(all_as)):
        bad_bit = bad_bits[torch.randperm(num_bad_bits)][0]
        a = (torch.rand(n) < pa).long()
        a[bad_bit] = 1
        b = (torch.rand(n) < pb).long()
        all_as.append(a)
        all_bs.append(b)

    all_as = torch.stack(all_as)    # (r,n)
    all_bs = torch.stack(all_bs)    # (r,n)
    all_rules = torch.cat([all_as, all_bs], dim=1) # (r,2n)

    if return_dict:
        return {
            "rules" : all_rules[torch.randperm(r),:],
            "bad_bits" : bad_bits
        }
    else:
        return all_rules[torch.randperm(r),:]


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


def compose_rules(rules1: torch.LongTensor, rules2: torch.LongTensor):
    """ Compose rules1 @> rules2
        Given: rules1 = {a1 -> b1}, rules2 = {a2 -> b2}
        Then:  rules3 = {a1 -> s1}, where s1 = rules2(rules1(a1)) for each a1
    """
    assert rules1.shape == rules2.shape
    _, r, _ = rules1.shape
    a, _ = split_abs(rules1)
    succs = []
    for k in range(r):
        ak = a[:,k,:] # (N,n)
        sk, _ = step_rules(rules1, ak)
        sk, _ = step_rules(rules2, sk)
        succs.append(sk)

    succs = torch.stack(succs, dim=1) # (N,r,n)
    comp_rules = torch.cat([a, succs], dim=2) # (N,r,2n)
    return comp_rules.long()  # (N,r,2n)


def prove_theorem(rules: torch.LongTensor, theorem: torch.LongTensor):
    """ Run a proof and return a bunch of metadata """
    N, r, n2 = rules.shape
    _, n = theorem.shape
    assert n2 == n*2

    all_states, all_hits, chain_len = [], [], torch.zeros(N).long()
    z = torch.zeros_like(theorem)
    for t in range(n):
        z_new, h = step_rules(rules, z)
        all_states.append(z_new)
        all_hits.append(h)
        chain_len += ((z_new - z).sum(dim=1)) > 0   # Incr if derived something new
        z = z_new

    return {
        "rules": rules,
        "theorem": theorem,
        "qed": all_leq(theorem, z),
        "states": torch.stack(all_states, dim=1), # (N,n,n)
        "hits": torch.stack(all_hits, dim=1), # (N,n,r)
        "chain_len": chain_len, # (N,)
    }

