import torch


def random_rules(N, r, n, ante_prob, conseq_prob):
    assert 0 < ante_prob and ante_prob < 1
    assert 0 < conseq_prob and conseq_prob < 1
    a = (torch.rand(N,r,n) < ante_prob).int()
    b = (torch.rand(N,r,n) < conseq_prob).int()
    rules = torch.cat([a, b], dim=2)
    return rules.int()  # (N,r,2n)


def antes_conseqs(rules):
    """ Split the antecedents and consequents
    """
    assert rules.ndim == 3 and rules.size(2) % 2 == 0
    return rules.chunk(2, dim=2)


def step_rules(rules, state, eps=1e-5):
    """ Apply one step of the rules
    """
    N, r, n2 = rules.shape
    _, n = state.shape
    assert n2 == 2*n and state.size(0) == N
    a, b = antes_conseqs(rules)         # (N,r,n), (N,r,n)
    hits = a <= state.view(N,1,n) + eps # (N,r,n)
    hits = n <= hits.sum(dim=2) + eps   # (N,r)

    # Original definition
    succ = (b * hits.view(N,r,1)).sum(dim=1)    # (N,n)
    succ = (state + succ).clamp(0,1)            # (N,n)
    return succ.int(), hits.int()   # (N,n), (N,r)


def compose_rules(rules1, rules2):
    """ Compose rules1 @> rules2
        Given: rules1 = {a1 -> b1}, rules2 = {a2 -> b2}
        Then:  rules3 = {a1 -> s1}, where s1 = rules2(rules1(a1)) for each a1
    """
    assert rules1.shape == rules2.shape and rules1.size(2) % 2 == 0
    _, r, _ = rules1.shape

    succs = []
    a, _ = antes_conseqs(rules1)
    for k in range(r):
        ak = a[:,k,:] # (N,n)
        sk, _ = step_rules(rules1, ak)
        sk, _ = step_rules(rules2, sk)
        succs.append(sk)

    succs = torch.stack(succs, dim=1) # (N,r,n)
    comp_rules = torch.cat([a, succs], dim=2) # (N,r,2n)
    return comp_rules.int()  # (N,r,2n)


