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
    state_prob: Optional[float] = None,
    num_fillers: int = 2,
    return_dict: bool = False
):
    """ Make one rule at a time (batch_size == 1) """
    assert num_rules > num_fillers + chain_len
    assert num_vars > chain_len
    r, n, ap, bp, sp = num_rules, num_vars, ante_prob, conseq_prob, state_prob

    order = torch.randperm(n)
    chain_bits = order[:chain_len]    # We need chain_len - 1 dedicated bits
    other_bits = order[chain_len:n-1] # The other stuffs
    bad_bit = order[-1] # The bad bit we wanna be excluding

    other_vec = torch.zeros(n)
    other_vec[other_bits] = 1

    init_state = torch.zeros(n)
    if sp is not None:
        init_state = (torch.rand(n) < sp)
        init_state[chain_bits] = 0
        init_state[bad_bit] = 0
    init_state = init_state.long()

    all_as, all_bs = [], []

    # Construct the chain first
    all_as.append((torch.rand(n) < ap) * init_state)
    all_bs.append(F.one_hot(chain_bits[0], n) + (torch.rand(n) < bp) * other_vec)
    for k in range(len(chain_bits) - 1):
        all_as.append(F.one_hot(chain_bits[k], n) + (torch.rand(n) < ap) * init_state)
        all_bs.append(F.one_hot(chain_bits[k+1], n) + (torch.rand(n) < bp) * other_vec)

    # Now make the filler rules
    for _ in range(num_fillers):
        all_as.append((torch.rand(n) < ap) * init_state)
        all_bs.append((torch.rand(n) < bp) * other_vec)

    # Now add the adversarial rules
    num_bad_rules = r - len(all_as)
    for _ in range(num_bad_rules):
        bad_a = (torch.rand(n) < ap).long()
        bad_a[bad_bit] = 1
        all_as.append(bad_a)
        all_bs.append(torch.rand(n) < bp)

    all_as = torch.stack(all_as)    # (r,n)
    all_bs = torch.stack(all_bs)    # (r,n)
    all_rules = torch.cat([all_as, all_bs], dim=1).long()
    all_rules = all_rules[torch.randperm(r)]

    if return_dict:
        return {
            "rules": all_rules,
            "init_state": init_state,
            "chain_bits": chain_bits,
            "other_bits": other_bits,
            "bad_bit" : bad_bit
        }
    else:
        return all_rules, init_state


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

    all_states, all_hits, chain_len = [], [], torch.zeros(N).long()
    z = torch.zeros_like(theorem) if init_state is None else init_state
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

