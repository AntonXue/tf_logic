import torch


def random_rules(N, r, n, ante_prob, conseq_prob, ensure_facts=True):
    assert 0 < ante_prob and ante_prob < 1
    assert 0 < conseq_prob and conseq_prob < 1

    # Guarantee at least one zero rule exists
    a = (torch.rand(N,r,n) < ante_prob).long()
    if ensure_facts:
        a[:,0:2,:] = torch.zeros(N,2,n).long()

    b = (torch.rand(N,r,n) < conseq_prob).long()
    rules = torch.cat([a, b], dim=2)
    return rules.long()  # (N,r,2n)


def antes_conseqs(rules):
    """ Split the antecedents and consequents
    """
    assert rules.ndim == 3 and rules.size(2) % 2 == 0
    return rules.chunk(2, dim=2)


def all_leq(x, y, eps=1e-5):
    """ Do leq testing on the last coordinate with tolerance eps
    """
    assert x.ndim == y.ndim
    n = x.size(-1)
    z = (x <= y + eps).sum(dim=-1) # (*,)
    return (n <= z + eps).long()


def step_rules(rules, state, eps=1e-5, inf_mode=None):
    """ Apply one step of the rules.
        Inference modes:
            'residual': s' = s + sum_{a,b} b * M(a,s)
            'ab_only':  s' = sum_{a,b} (a+b) * M(a,s)
    """
    N, r, n2 = rules.shape
    _, n = state.shape
    assert n2 == n*2
    a, b = antes_conseqs(rules)         # (N,r,n), (N,r,n)
    hits = all_leq(a, state.view(N,1,n), eps=eps) # (N,r)

    # hits = a <= state.view(N,1,n) + eps # (N,r,n)
    # hits = n <= hits.sum(dim=2) + eps   # (N,r)

    # Original definition
    if inf_mode is None or inf_mode == "residual":
        succ = (b * hits.view(N,r,1)).sum(dim=1)    # (N,n)
        succ = (state + succ).clamp(0,1)            # (N,n)
    elif inf_mode == "ab_only":
        succ = ((a + b) * hits.view(N,r,1)).sum(dim=1).clamp(0,1)   # (N,1)
    else:
        raise NotImplementedError()

    return succ.long(), hits.long()   # (N,n), (N,r)


def compose_rules(rules1, rules2, inf_mode=None):
    """ Compose rules1 @> rules2
        Given: rules1 = {a1 -> b1}, rules2 = {a2 -> b2}
        Then:  rules3 = {a1 -> s1}, where s1 = rules2(rules1(a1)) for each a1
    """
    assert rules1.shape == rules2.shape
    _, r, _ = rules1.shape

    succs = []
    a, _ = antes_conseqs(rules1)
    for k in range(r):
        ak = a[:,k,:] # (N,n)
        sk, _ = step_rules(rules1, ak, inf_mode=inf_mode)
        sk, _ = step_rules(rules2, sk, inf_mode=inf_mode)
        succs.append(sk)

    succs = torch.stack(succs, dim=1) # (N,r,n)
    comp_rules = torch.cat([a, succs], dim=2) # (N,r,2n)
    return comp_rules.long()  # (N,r,2n)


def prove_theorem(rules, theorem, eps=1e-5, inf_mode=None):
    """ Run a proof and return a bunch of metadata
    """
    N, r, n2 = rules.shape
    _, n = theorem.shape
    assert n2 == n*2

    # history
    zs, hs = [], []
    z = torch.zeros_like(theorem)
    for t in range(n):
        z, h = step_rules(rules, z, eps=eps, inf_mode=inf_mode)
        zs.append(z)
        hs.append(h)

    zs = torch.stack(zs, dim=1) # (N,n,n)
    hs = torch.stack(hs, dim=1) # (N,n,r)
    qed = all_leq(theorem, z, eps=eps)
    return {
        "rules" : rules,
        "theorem" : theorem,
        "qed" : qed,
        "inf_mode" : inf_mode,
        "states" : zs,
        "hists" : hs
    }


