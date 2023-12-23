def stringify_rule(rule, var_sep_token):
    """
    Create a rule of the form xi , xj , ... -> xa
    from a one-hot vector of [<ants>, <cons>]
    """

    n_vars = len(rule) // 2
    ants = [f"x{i}" for i in range(n_vars) if rule[i]]
    cons = [f"x{i}" for i in range(n_vars) if rule[n_vars+i]]
    if len(ants) < 1:
        ants = ["empty"]
    if len(cons) < 1:
        cons = ["empty"]
    rule = var_sep_token.join(ants) + " -> " + var_sep_token.join(cons)
    return rule

def get_string_rep(dataset_item):
    """
    Returns a string of the form:
    [RULES_START] [RULE_START] ... [RULE_END] ... [RULES_END]
    [THEOREM_START] ... [THEOREM_END]
    [QED]
    """

    # Define the placeholder tokens
    var_sep_token = " , "
    rules_start = "[RULES_START]"
    rules_end = "[RULES_END]"
    rule_start = "[RULE_START]"
    rule_end = "[RULE_END]"
    theorem_start = "[THEOREM_START]"
    theorem_end = "[THEOREM_END]"
    qed = "[QED]"

    rules = dataset_item["rules"]
    theorem = dataset_item["theorem"]

    n_vars = len(theorem)

    rule_strs = [rule_start + " " + stringify_rule(rule, var_sep_token) + " " + rule_end for rule in rules]
    theorem_str = var_sep_token.join([f"x{i}" for i in range(n_vars) if theorem[i]])
    theorem_str = theorem_start + " " + theorem_str + " " + theorem_end
    rules_str = rules_start + " " + " ".join(rule_strs) + " " + rules_end
    return rules_str + " " + theorem_str + " " + qed
