import torch
from torch.utils.data import Dataset
import os
import json
import itertools
import random

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
        self.dataset = self.generate_dataset()
        # Shuffle the dataset
        random.seed(seed)
        random.shuffle(self.dataset)
        self.dataset_len = min(len(self.dataset), dataset_len)

    def __len__(self):
        return self.dataset_len

    def _get_minecraft_recipes(self):
        dataset_dir = r"/home/akhare/repos/Minecraft-Crafting-Web/minecraft-data/recipes"
        recipe_files = os.listdir(dataset_dir)
        recipe_files = [os.path.join(dataset_dir, f) for f in recipe_files if f.endswith(".json")]
        recipes = [json.load(open(f)) for f in recipe_files]
        return recipes

    def _get_minecraft_rules(self, recipes):
        rules = []
        for recipe in recipes:
            try:
                if "crafting" not in recipe["type"]:
                    continue
                antecedents = set()
                consequents = set()
                antecedents_group_key = "key" if "key" in recipe else "ingredients" if "ingredients" in recipe else "ingredient" if ("ingredient" in recipe and len(recipe["ingredient"]) > 1) else None
                if antecedents_group_key is not None:
                    for ingredient in (recipe[antecedents_group_key].values() if antecedents_group_key == "key" else recipe[antecedents_group_key]):
                        antecedents.add(ingredient["item"] if "item" in ingredient else ingredient["tag"])
                else:
                    if "ingredient" in recipe:
                        antecedents.add(recipe["ingredient"]["item"] if "item" in recipe["ingredient"] else recipe["ingredient"]["tag"])
                    if "base" in recipe:
                        antecedents.add(recipe["base"]["item"] if "item" in recipe["base"] else recipe["base"]["tag"])
                    if "addition" in recipe:
                        antecedents.add(recipe["addition"]["item"] if "item" in recipe["addition"] else recipe["addition"]["tag"])
                if "result" in recipe:
                    consequents.add(recipe["result"]["item"] if "item" in recipe["result"] else recipe["result"])
                if len(consequents) > 0:
                    rules.append((antecedents, consequents, recipe))
            except:
                continue
        return rules
    
    def _get_all_ways_to_craft(self, item, rules, items_so_far=None, depth=0):
        """Get all the ways to craft an item

        Args:
            item (str): The item to craft
            rules (list): List of all rules
            items_so_far (list, optional): Items already crafted. Defaults to None.
            depth (int, optional): Depth of the chain. Defaults to 0.

        Returns:
            histories: List of all ways to craft an item
        """
        histories = []
        for antecedents, consequents, _ in rules:
            if item in consequents:
                history = [(antecedents, item, depth)]
                if items_so_far is None:
                    items_so_far = [item]
                else:
                    items_so_far.append(item)
                antecedent_histories = []
                for antecedent in antecedents:
                    if antecedent in items_so_far:
                        continue
                    all_ways_to_craft_antecedent = self._get_all_ways_to_craft(antecedent, rules, items_so_far, depth + 1)
                    if len(all_ways_to_craft_antecedent) > 0:
                        antecedent_histories.append(all_ways_to_craft_antecedent)
                histories.append(history)
                
                for antecendent_way_combination in itertools.product(*antecedent_histories):
                    for chain_len in range(1, len(antecendent_way_combination) + 1):
                        for antecedent_history_combination in itertools.combinations(antecendent_way_combination, chain_len):
                            history_combination = history.copy()
                            # Extend by flattening the list
                            for antecedent_history in antecedent_history_combination:
                                history_combination.extend(antecedent_history)
                            histories.append(history_combination)
        return histories
    
    def _remove_redundant_histories(self, histories):
        """Prune the list of histories to only unique histories
        
        Args:
            histories (list): List of histories
            
        Returns:
            unique_histories: List of unique histories
        """
        unique_histories = []
        for history in histories:
            if history not in unique_histories:
                unique_histories.append(history)
        return unique_histories

    def generate_dataset(self):
        recipes = self._get_minecraft_recipes()
        rules = self._get_minecraft_rules(recipes)
        all_items = set(list(rule[1])[0] for rule in rules)
        all_histories = []
        for item in all_items:
            ways_to_craft_item = self._get_all_ways_to_craft(item, rules)
            for way in ways_to_craft_item:
                # Check if depth < num_steps
                if max([step[2] for step in way]) < self.num_steps:
                    continue
                # Cut off the history at num_steps
                way_steps = [step for step in way if step[2] <= self.num_steps]
                # Add distractor rules
                num_distractors = random.randint(0, self.max_num_distractors)
                distractor_rules = random.sample(rules, num_distractors)
                for distractor_rule in distractor_rules:
                    way_steps.append((distractor_rule[0], distractor_rule[1], -1))
                # Add the history to the list of all histories
                all_histories.append(way_steps)

        print("Number of histories: ", len(all_histories))
        print("Number of unique histories: ", len(self._remove_redundant_histories(all_histories)))
        return all_histories
        # return self._remove_redundant_histories(all_histories)  # Can result in variable length dataset since distractors are randomly added
    
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

    def _get_string_rep(self, recipe: list, qed: bool = True):
        true_facts = []
        distractor_facts = []
        facts = []
        target = None
        rules = []
        max_depth = max([rule[2] for rule in recipe])
        rules_str = "[RULES_START] "
        if not qed and len(recipe) > 1:
            # Choose a random subset of rules
            target_rule = recipe[0]
            recipe = random.sample(recipe[1:], random.randint(1, len(recipe)-1))
            recipe.append(target_rule)
        random.shuffle(recipe)
        for rule in recipe:
            if rule[2] == 0:
                target = rule[1]
                # Check if any of the antecedents are facts
                for antecedent in rule[0]:
                    if antecedent not in [rule[1] for rule in recipe]:
                        true_facts.append(antecedent)
            elif rule[2] == max_depth:
                true_facts.extend(rule[0])
            elif rule[2] == -1:
                distractor_facts.extend(rule[0])
            if (rule[0], rule[1]) not in rules:
                rules.append((rule[0], rule[1]))
                rules_str += f"{' + '.join(rule[0])} -> {''.join(rule[1])} , "
        facts = true_facts + distractor_facts
        if not qed:
            # Choose a random subset of facts
            if len(true_facts) > 0:
                true_facts = random.sample(true_facts, random.randint(0, len(true_facts)-1))
            if len(distractor_facts) > 0:
                distractor_facts = random.sample(distractor_facts, random.randint(0, len(distractor_facts)))
            facts = true_facts + distractor_facts
        # Shuffle the facts
        random.shuffle(facts)
        rules_str = rules_str[:-2]
        rules_str += "[RULES_END]"
        facts_str = "[FACTS_START] " + " , ".join(set(facts)) + " [FACTS_END]"
        target_str = f"[TARGET_START] {target} [TARGET_END]"
        final_str = f"{rules_str} {facts_str} {target_str}"
        # Remove minecraft: prefix
        final_str = final_str.replace("minecraft:", "")
        return final_str
    
    def __getitem__(self, idx):
        recipe = self.dataset[idx]
        qed = True if idx % 2 == 0 else False
        if qed:
            labels = torch.tensor(1).long()
        else:
            labels = torch.tensor(0).long()
        item = self._get_string_rep(recipe, qed)
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