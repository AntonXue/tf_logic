import torch


def random_rules(N, r, n, ante_prob, conseq_prob):
    assert 0 < ante_prob and ante_prob < 1
    assert 0 < conseq_prob and conseq_prob < 1
    antes = (torch.rand(N,r,n) < ante_prob).int()
    conseqs = (torch.rand(N,r,n) < conseq_prob).int()
    rules = torch.cat([antes, conseqs], dim=2)
    return rules.int()  # (N,r,2n)


def antes_conseqs(rules):
    """ Split the antecedents and consequents
    """
    assert rules.ndim == 3 and rules.size(2) % 2 == 0
    return rules.chunk(2, dim=2)


def one_step(rules, state, eps=1e-5):
    """ Apply one step of the rules
    """
    N, r, n2 = rules.shape
    assert state.shape == torch.Size([N,n2//2]) and n2 % 2 == 0
    n = n2 // 2
    antes, conseqs = antes_conseqs(rules)
    hits = (antes <= state.view(N,1,n) + eps)   # (N,r,n)
    hits = (hits.sum(dim=2) <= n + eps)     # (N,r)

    succ = (conseqs * hits.view(N,r,1)) # (N,r,n)
    succ = succ.sum(dim=1).clamp(0,1) # (N,n)
    return succ.int(), hits.int()   # (N,n), (N,r)


def compose_rules(rules1, rules2, eps=1e-5):
    """ Compose rules1 @> rules2
    """
    assert rules1.shape == rules2.shape and rules1.size(2) % 2 == 0
    N, r, n2 = rules1.shape
    n = n2 // 2
    s, _ = one_step(rules1, torch.zeros(N,n).to(rules1.device), eps=eps)

    antes2, conseqs2 = antes_conseqs(rules2)
    new_antes2 = (antes2 - s.view(N,1,n)).clamp(0,1)    # (N,r,n)
    new_rules = torch.cat([new_antes2, conseqs2], dim=2)
    return new_rules.int()  # (N,r,2n)



