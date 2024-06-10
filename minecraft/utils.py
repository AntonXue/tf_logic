"""
Dataset utils.
"""
import random

def stringify_recipe_with_tags(
        recipe: dict,
        task_type: str,
        cot_states: list,
        rule_start_tag: str = "[RULES_START]",
        rule_end_tag: str = "[RULES_END]",
        fact_start_tag: str = "[FACTS_START]",
        fact_end_tag: str = "[FACTS_END]",
        target_start_tag: str = "[TARGET_START]",
        target_end_tag: str = "[TARGET_END]",
        rules_separator: str = " , ",
        facts_separator: str = " , ",
        rules_ante_conseq_separator: str = " -> ",
        rules_ante_separator: str = " + ",
        cot_states_start_tag: str = "[STATES_START]",
        cot_states_end_tag: str = "[STATES_END]",
        cot_states_separator: str = " , ",
        include_antecedents_in_cot_states: bool = True,
        shuffle: bool = True
    ):
        """Convert the components of a recipe to a string representation with tags.

        Args:
            recipe (dict): The components of the recipe.
            task_type (str): The type of task. Supported task types: binary_classification, next_token_prediction
            cot_states (list): The list of COT states.
            rule_start_tag (str, optional): The start tag for the rules. Defaults to "[RULES_START]".
            rule_end_tag (str, optional): The end tag for the rules. Defaults to "[RULES_END]".
            fact_start_tag (str, optional): The start tag for the facts. Defaults to "[FACTS_START]".
            fact_end_tag (str, optional): The end tag for the facts. Defaults to "[FACTS_END]".
            target_start_tag (str, optional): The start tag for the target. Defaults to "[TARGET_START]".
            target_end_tag (str, optional): The end tag for the target. Defaults to "[TARGET_END]".
            rules_separator (str, optional): The separator for the rules. Defaults to " , ".
            facts_separator (str, optional): The separator for the facts. Defaults to " , ".
            rules_ante_conseq_separator (str, optional): The separator between the antecedent and consequent of a rule. Defaults to " -> ".
            rules_ante_separator (str, optional): The separator between the antecedent elements of a rule. Defaults to " + ".
            cot_states_start_tag (str, optional): The start tag for the COT states. Defaults to "[STATES_START]".
            cot_states_end_tag (str, optional): The end tag for the COT states. Defaults to "[STATES_END]".
            cot_states_separator (str, optional): The separator for the COT states. Defaults to " , ".
            include_antecedents_in_cot_states (bool, optional): Whether to include the antecedents in the COT states. Defaults to True.
            shuffle (bool, optional): Whether to shuffle the facts and rules. Defaults to True.

        Returns:
            str: String representation of the recipe (<rules_start_tag> <rules> <rules_end_tag> <facts_start_tag> <facts> <facts_end_tag> <target_start_tag> <target> <target_end_tag>)
        """
        if shuffle:
            random.shuffle(recipe["rules"])
            random.shuffle(recipe["facts"])

        rules = rules_separator.join(
            [
                f"{rules_ante_separator.join(rule[0])} {rules_ante_conseq_separator} {rule[1]}"
                for rule in recipe["rules"]
            ]
        )
        facts = facts_separator.join(recipe["facts"])
        if task_type == "binary_classification":
            target = recipe["target"]
            final_str = f"{rule_start_tag} {rules} {rule_end_tag} {fact_start_tag} {facts} {fact_end_tag} {target_start_tag} {target} {target_end_tag}"
        elif task_type == "next_token_prediction":
            states = cot_states_separator.join([state["fact"] for state in cot_states])
            if include_antecedents_in_cot_states:
                 states = cot_states_separator.join(
                      [
                           f"{rules_ante_separator.join(state['antecedents'])} {rules_ante_conseq_separator} {state['fact']}"
                            for state in cot_states
                      ]
                 )
            final_str = f"{rule_start_tag} {rules} {rule_end_tag} {fact_start_tag} {facts} {fact_end_tag} {cot_states_start_tag} {states} {cot_states_end_tag}"
        else:
            raise Exception("Task type not supported. Supported task types: binary_classification, next_token_prediction")
        return final_str.replace("minecraft:", "")

def stringify_recipe_with_text(
        recipe: dict,
        task_type: str,
        cot_states: list,
        shuffle: bool = True,
        depth_parallel: bool = False,
        return_states: bool = False
):
    """Convert the components of a recipe to a string representation with text.
    The text representation is as follows:
    - For binary_classification task type: (Rules) If I have x1, x2 and x3, then I can create y1. (Facts) I have x1, x2 and x3. (Target) Can I create y1?
    - For next_token_prediction task type: (Rules) If I have x1, x2 and x3, then I can create y1. (Facts) I have x1, x2 and x3. (States) I have x1, x2 and x3 and so I can create y1.

    Args:
        recipe (dict): The components of the recipe.
        task_type (str): The type of task. Supported task types: binary_classification, next_token_prediction
        cot_states (list): The list of COT states with each item being a dict with antecedents and fact.
        shuffle (bool, optional): Whether to shuffle the facts and rules. Defaults to True.

    Returns:
        str: String representation of the recipe
    """
    if shuffle:
        random.shuffle(recipe["rules"])
        random.shuffle(recipe["facts"])

    rules = " ".join(
        [
            f"If I have {' and '.join(rule[0])}, then I can create {rule[1]}."
            for rule in recipe["rules"]
        ]
    )

    facts = f"I have {' and '.join(recipe['facts'])}."

    if task_type == "binary_classification":
        target = f"Can I create {recipe['target']}?"
        final_str = f"{rules}\n{facts}\n{target}"
    elif task_type == "next_token_prediction":
        states = " ".join(
            [
                f"I have {' and '.join(state['antecedents'])} and so I can create {state['fact']}."
                for state in cot_states
            ]
        )
        if depth_parallel:
            states = ""
            if len(cot_states) > 0:
                reasoning_depth = cot_states[-1]["depth"]
                for d in range(reasoning_depth+1):
                    derivations_at_d = [state for state in cot_states if state["depth"] == d]
                    antecedents_at_d = []
                    facts_at_d = []
                    for state in derivations_at_d:
                        antecedents_at_d.extend(state['antecedents'])
                        facts_at_d.append(state['fact'])
                    states += f"I have {' and '.join(antecedents_at_d)} and so I can create {' and '.join(facts_at_d)}."
                    states += " "
                states = states.strip()
        # final_str = f"{rules}\n{facts}\n{states}".replace("_", " ")
        final_str = f"""Here are some crafting recipes: \n{rules}\nHere are some items I have: \n{facts}\nBased on the items I have and the crafting recipes, I can create the following items: \n{states}\nI cannot create any other items."""
        final_str = final_str.replace("_", " ")
    else:
        raise Exception("Task type not supported. Supported task types: binary_classification, next_token_prediction")
    
    if return_states:
        return final_str, states
    return final_str.replace("minecraft:", "")



