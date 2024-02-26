import torch
from torch.utils.data import Dataset
import os
import json
import itertools
import random

"""
TODO: Add support for num_items range
"""

class MinecraftAutoregKStepsTokensDataset(Dataset):
    def __init__(
        self,
        num_steps: int,
        dataset_len: int,
        max_num_distractors: int = 2,
        seed: int = 101,
        tokenizer = None,
        padding: str = "longest",
    ):
        self.num_steps = num_steps - 1  # Since the first step is the target (depth = 0)
        self.seed = seed
        self.max_num_distractors = max_num_distractors
        self.tokenizer = tokenizer
        self.padding = padding
        self.dataset_len = dataset_len
        self.dataset = self._generate_base_dataset()
        self.dataset = self._generate_dataset_with_distractions()
        # Shuffle the dataset
        random.seed(seed)
        random.shuffle(self.dataset)

    def __len__(self):
        return self.dataset_len

    def _get_minecraft_rules(self):
        dataset_dir = r"/home/akhare/repos/Minecraft-Crafting-Web/minecraft-data/recipes"
        raw_rule_files = os.listdir(dataset_dir)
        raw_rule_files = [os.path.join(dataset_dir, f) for f in raw_rule_files if f.endswith(".json")]
        raw_rules = [json.load(open(f)) for f in raw_rule_files]
        return raw_rules

    def _process_minecraft_rules(self, raw_rules):
        """Collect all the crafting rules from the raw rules.

        Args:
            raw_rules (list): List of raw rules

        Returns:
            rules: List of crafting rules where each elemet is a (antecedents, consequents, raw_rule) tuple
        """
        rules = []
        for raw_rule in raw_rules:
            try:
                if "crafting" not in raw_rule["type"]:
                    continue
                antecedents = set()
                consequents = set()
                antecedents_group_key = "key" if "key" in raw_rule else "ingredients" if "ingredients" in raw_rule else "ingredient" if ("ingredient" in raw_rule and len(raw_rule["ingredient"]) > 1) else None
                if antecedents_group_key is not None:
                    for ingredient in (raw_rule[antecedents_group_key].values() if antecedents_group_key == "key" else raw_rule[antecedents_group_key]):
                        antecedents.add(ingredient["item"] if "item" in ingredient else ingredient["tag"])
                else:
                    if "ingredient" in raw_rule:
                        antecedents.add(raw_rule["ingredient"]["item"] if "item" in raw_rule["ingredient"] else raw_rule["ingredient"]["tag"])
                    if "base" in raw_rule:
                        antecedents.add(raw_rule["base"]["item"] if "item" in raw_rule["base"] else raw_rule["base"]["tag"])
                    if "addition" in raw_rule:
                        antecedents.add(raw_rule["addition"]["item"] if "item" in raw_rule["addition"] else raw_rule["addition"]["tag"])
                if "result" in raw_rule:
                    consequents.add(raw_rule["result"]["item"] if "item" in raw_rule["result"] else raw_rule["result"])
                if len(consequents) > 0:
                    rules.append((antecedents, consequents, raw_rule))
            except:
                continue
        return rules
    
    def _get_all_recipes_for_item(self, item, rules, items_so_far=None, depth=0):
        """Get all the recipes for an item

        Args:
            item (str): The item to craft
            rules (list): List of all rules
            items_so_far (list, optional): Items already crafted. Defaults to None.
            depth (int, optional): Depth of the chain. Defaults to 0.

        Returns:
            recipes: List of all recipes for an item (all ways to craft an item)
        """
        recipes = []
        for antecedents, consequents, _ in rules:
            if item in consequents:
                recipe = [(antecedents, item, depth)]
                if items_so_far is None:
                    items_so_far = [item]
                else:
                    items_so_far.append(item)
                antecedent_recipes = []
                for antecedent in antecedents:
                    if antecedent in items_so_far:
                        continue
                    all_recipes_for_antecedent = self._get_all_recipes_for_item(antecedent, rules, items_so_far, depth + 1)
                    if len(all_recipes_for_antecedent) > 0:
                        antecedent_recipes.append(all_recipes_for_antecedent)
                recipes.append(recipe)
                
                for antecendent_recipe_combination in itertools.product(*antecedent_recipes):
                    for chain_len in range(1, len(antecendent_recipe_combination) + 1):
                        for antecedent_recipe_combination_at_chain_len in itertools.combinations(antecendent_recipe_combination, chain_len):
                            recipe_combination = recipe.copy()
                            # Extend by flattening the list
                            for antecedent_recipe in antecedent_recipe_combination_at_chain_len:
                                recipe_combination.extend(antecedent_recipe)
                            if recipe_combination not in recipes:
                                recipes.append(recipe_combination)
        return recipes
    
    def _remove_redundant_recipes(self, recipes):
        """Prune the list of recipes to only unique recipes
        
        Args:
            recipes (list): List of recipes
            
        Returns:
            unique_recipes: List of unique recipes
        """
        unique_recipes = []
        for recipe in recipes:
            if recipe not in unique_recipes:
                unique_recipes.append(recipe)
        return unique_recipes
    
    def _get_components(self, recipe: list):
        """Get the components of a recipe
        Components are rules (antecedents, consequents) pairs, facts and the target

        Args:
            recipe (list): The recipe

        Returns:
            components (dict): Dictionary containing the components (rules, facts, target)
        """

        rules = [(rule[0], rule[1]) for rule in recipe]
        derivables = [rule [1] for rule in recipe]
        facts = [antecedent for rule in recipe for antecedent in rule[0] if antecedent not in derivables]
        target = [rule[1] for rule in recipe if rule[2] == 0][0]

        return {
            "rules": rules,
            "facts": facts,
            "target": target
        }
    
    def _get_components_with_distrations(self, recipe: list, qed: bool = True):
        """Get the components of a recipe with distractors.
        Distractors are recipes that can be derived but are not the target.

        Args:
            recipe (list): The recipe
            qed (bool, optional): Whether the recipe is qed or not. Defaults to True.

        Returns:
            components (dict): Dictionary containing the components (rules, facts, target)
        """
        
        recipe_components = self._get_components(recipe)
        num_distractors = random.randint(0, self.max_num_distractors)
        # Distractors are recipes that can be derived but are not the target
        distractor_recipes = random.sample(self.dataset, num_distractors)
        distractor_components = [self._get_components(distractor_recipe) for distractor_recipe in distractor_recipes]

        if qed:
            return {
                "rules": recipe_components["rules"] + [distractor["rules"][0] for distractor in distractor_components],
                "facts": recipe_components["facts"] + [antecedent for distractor in distractor_components for antecedent in distractor["rules"][0][0]],
                "target": recipe_components["target"],
                "qed": True
            }
        
        # If not qed, choose a random subset of rules and facts (cannot include all facts)
        return {
            "rules": random.sample(recipe_components["rules"], random.randint(0, len(recipe_components["rules"]))) + [distractor["rules"][0] for distractor in distractor_components],
            "facts": random.sample(recipe_components["facts"], random.randint(0, len(recipe_components["facts"])-1)) + [antecedent for distractor in distractor_components for antecedent in distractor["rules"][0][0]],
            "target": recipe_components["target"],
            "qed": False
        }
    
    def _generate_base_dataset(self):
        """Generate the base dataset.

        Returns:
            dataset: List of all recipes with num_steps steps
        """

        raw_rules = self._get_minecraft_rules()
        rules = self._process_minecraft_rules(raw_rules)
        all_items = set(list(rule[1])[0] for rule in rules)
        all_recipes = []
        for item in all_items:
            all_recipes_for_item = self._get_all_recipes_for_item(item, rules)
            for recipe in all_recipes_for_item:
                # Check if depth < num_steps
                if max([rule[2] for rule in recipe]) < self.num_steps:
                    continue
                # Cut off the recipe at num_steps
                if self.num_steps >= 0:
                    recipe_with_num_steps = [rule for rule in recipe if rule[2] <= self.num_steps]
                else:
                    recipe_with_num_steps = recipe
                # Add the recipe to the list of all recipes
                all_recipes.append(recipe_with_num_steps)

        print("Number of recipes: ", len(all_recipes))
        print("Number of unique recipes: ", len(self._remove_redundant_recipes(all_recipes)))
        return all_recipes
    
    def _generate_dataset_with_distractions(self):
        """Generate the dataset with distractors.
        The dataset is generated by taking the original dataset and adding distractors to it.
        The size of the new dataset matches the requirement by creating multiple copies of the samples from the original dataset and randomly adding distractions.
        The dataset is balanced with half the samples being qed and the other half being non-qed.
        """

        dataset = []
        num_samples_per_recipe = self.dataset_len // len(self.dataset)
        num_qed_samples = num_samples_per_recipe // 2
        for recipe in self.dataset:
            qed_samples = [self._get_components_with_distrations(recipe, qed=True) for _ in range(num_qed_samples)]
            non_qed_samples = [self._get_components_with_distrations(recipe, qed=False) for _ in range(num_samples_per_recipe - num_qed_samples)]
            dataset.extend(qed_samples + non_qed_samples)
        return dataset
    
    def _stringify_recipe(self, recipe: dict, rule_start_tag: str = "[RULES_START]", rule_end_tag: str = "[RULES_END]", fact_start_tag: str = "[FACTS_START]", fact_end_tag: str = "[FACTS_END]", target_start_tag: str = "[TARGET_START]", target_end_tag: str = "[TARGET_END]", rules_separator: str = " , ", facts_separator: str = " , ", rules_ante_conseq_separator: str = " -> ", rules_ante_separator: str = " + "):
        """Convert the components of a recipe to a string representation.

        Args:
            recipe (dict): The components of the recipe
            rule_start_tag (str, optional): Start tag for rules. Defaults to "[RULES_START]".
            rule_end_tag (str, optional): End tag for rules. Defaults to "[RULES_END]".
            fact_start_tag (str, optional): Start tag for facts. Defaults to "[FACTS_START]".
            fact_end_tag (str, optional): End tag for facts. Defaults to "[FACTS_END]".
            target_start_tag (str, optional): Start tag for target. Defaults to "[TARGET_START]".
            target_end_tag (str, optional): End tag for target. Defaults to "[TARGET_END]".
            rules_separator (str, optional): Separator for rules. Defaults to " , ".
            facts_separator (str, optional): Separator for facts. Defaults to " , ".
            rules_ante_conseq_separator (str, optional): Separator for antecedents and consequents in rules. Defaults to " -> ".
            rules_ante_separator (str, optional): Separator for antecedents in rules. Defaults to " + ".

        Returns:
            str: String representation of the recipe (<rules_start_tag> <rules> <rules_end_tag> <facts_start_tag> <facts> <facts_end_tag> <target_start_tag> <target> <target_end_tag>)
        """

        rules = rules_separator.join([f"{rules_ante_separator.join(rule[0])} {rules_ante_conseq_separator} {rule[1]}" for rule in recipe["rules"]])
        facts = facts_separator.join(recipe["facts"])
        target = recipe["target"]
        final_str = f"{rule_start_tag} {rules} {rule_end_tag} {fact_start_tag} {facts} {fact_end_tag} {target_start_tag} {target} {target_end_tag}"
        return final_str.replace("minecraft:", "")
    
    def __getitem__(self, idx):
        recipe = self.dataset[idx]
        qed = recipe["qed"]
        if qed:
            labels = torch.tensor(1).long()
        else:
            labels = torch.tensor(0).long()
        item = self._stringify_recipe(recipe)
        if not self.tokenizer:
            return Exception("Tokenizer not provided.")
        encoding = self.tokenizer(item, truncation=True, padding=self.padding)

        return {
            "data": item,
            "labels": labels,
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "recipe": recipe,
        }