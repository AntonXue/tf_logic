import torch
from torch.utils.data import Dataset
import os
import json
import itertools
import random
from .utils import stringify_recipe_with_tags, stringify_recipe_with_text

"""
TODO: Add support for disjoint train and test datasets

Note: The getitem method will always return the same item for a given idx but the order of rules and facts will be shuffled.
"""

class MinecraftAutoregKStepsBaseDataset(Dataset):
    def __init__(
        self,
        num_steps: int,
        dataset_len: int,
        max_num_distractors: int = 2,
        seed: int = 101,
        tokenizer=None,
        padding: str = "longest",
        task_type: str = "binary_classification",    # Binary classification or Next Token Prediction
        example_format: str = "text",                 # Text or Tags  
        shuffle: bool = True
    ):
        assert task_type in ["binary_classification", "next_token_prediction"], "Task type should be either 'binary_classification' or 'next_token_prediction'"
        assert example_format in ["text", "tags"], "Example format should be either 'text' or 'tags'"
        self.task_type = task_type
        self.example_format = example_format
        self.num_steps = num_steps - 1  # Since the first step is the target (depth = 0)
        self.seed = seed
        self.max_num_distractors = max_num_distractors
        self.tokenizer = tokenizer
        self.padding = padding
        self.dataset_len = dataset_len
        self.dataset = self._generate_base_dataset()
        # Retain a copy of the original dataset
        self.base_dataset = self.dataset.copy()
        self.dataset = self._generate_dataset_with_distractions()
        self.dataset_len = len(self.dataset)
        # Shuffle the dataset
        if shuffle:
            random.seed(seed)
            random.shuffle(self.dataset)

    def __len__(self):
        return self.dataset_len

    def _get_minecraft_rules(self):
        dataset_dir = (
            r"/home/akhare/repos/Minecraft-Crafting-Web/minecraft-data/recipes"
        )
        raw_rule_files = os.listdir(dataset_dir)
        raw_rule_files = [
            os.path.join(dataset_dir, f) for f in raw_rule_files if f.endswith(".json")
        ]
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
                antecedents_group_key = (
                    "key"
                    if "key" in raw_rule
                    else "ingredients"
                    if "ingredients" in raw_rule
                    else "ingredient"
                    if ("ingredient" in raw_rule and len(raw_rule["ingredient"]) > 1)
                    else None
                )
                if antecedents_group_key is not None:
                    for ingredient in (
                        raw_rule[antecedents_group_key].values()
                        if antecedents_group_key == "key"
                        else raw_rule[antecedents_group_key]
                    ):
                        antecedents.add(
                            ingredient["item"]
                            if "item" in ingredient
                            else ingredient["tag"]
                        )
                else:
                    if "ingredient" in raw_rule:
                        antecedents.add(
                            raw_rule["ingredient"]["item"]
                            if "item" in raw_rule["ingredient"]
                            else raw_rule["ingredient"]["tag"]
                        )
                    if "base" in raw_rule:
                        antecedents.add(
                            raw_rule["base"]["item"]
                            if "item" in raw_rule["base"]
                            else raw_rule["base"]["tag"]
                        )
                    if "addition" in raw_rule:
                        antecedents.add(
                            raw_rule["addition"]["item"]
                            if "item" in raw_rule["addition"]
                            else raw_rule["addition"]["tag"]
                        )
                if "result" in raw_rule:
                    consequents.add(
                        raw_rule["result"]["item"]
                        if "item" in raw_rule["result"]
                        else raw_rule["result"]
                    )
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
            # If any antececent is in items so far, skip
            if items_so_far is not None and any(
                antecedent in items_so_far for antecedent in antecedents
            ):
                continue
            if item in consequents:
                recipe = [(antecedents, item, depth)]
                if items_so_far is None:
                    items_so_far = [item]
                else:
                    items_so_far.append(item)
                antecedent_recipes = []
                for antecedent in antecedents:
                    if antecedent in items_so_far:
                        # print("Antecedent already in items so far: ", antecedent)
                        continue
                    all_recipes_for_antecedent = self._get_all_recipes_for_item(
                        antecedent, rules, items_so_far, depth + 1
                    )
                    if len(all_recipes_for_antecedent) > 0:
                        antecedent_recipes.append(all_recipes_for_antecedent)
                recipes.append(recipe)

                for antecendent_recipe_combination in itertools.product(
                    *antecedent_recipes
                ):
                    for chain_len in range(1, len(antecendent_recipe_combination) + 1):
                        for (
                            antecedent_recipe_combination_at_chain_len
                        ) in itertools.combinations(
                            antecendent_recipe_combination, chain_len
                        ):
                            recipe_combination = recipe.copy()
                            # Extend by flattening the list
                            for (
                                antecedent_recipe
                            ) in antecedent_recipe_combination_at_chain_len:
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

    def _remove_loops(self, recipes):
        """Remove loops from the list of recipes.
        Loops are cases where there are rules in a recipe that can be derived from each other.

        Args:
            recipes (list): List of recipes

        Returns:
            recipes: List of recipes with no loops
        """
        for recipe in recipes:
            for rule in recipe:
                antecedents = rule[0]
                consequent = rule[1]
                depth = rule[2]
                for other_rule in recipe:
                    if other_rule[1] in antecedents and other_rule[2] <= depth:
                        print("----" * 10)
                        print("Found loop: ", rule, other_rule)
                        print(f"Recipe before: {recipe}")
                        print("----" * 10)
                        recipe.remove(other_rule)
            # If the recipe doesn't have depth 0, remove it
            # TODO: Move this to a higher level later
            if not any(rule[2] == 0 for rule in recipe):
                recipes.remove(recipe)
        return recipes

    def _get_components(self, recipe: list):
        """Get the components of a recipe
        Components are rules (antecedents, consequents) pairs, facts and the target

        Args:
            recipe (list): The recipe

        Returns:
            components (dict): Dictionary containing the components (rules, facts, target)
        """

        rules = [(rule[0], rule[1]) for rule in recipe]
        derivables = [rule[1] for rule in recipe]
        facts = [
            antecedent
            for rule in recipe
            for antecedent in rule[0]
            if antecedent not in derivables
        ]
        target = [rule[1] for rule in recipe if rule[2] == 0][0]
        depth = max(rule[2] for rule in recipe)

        return {"rules": rules, "facts": facts, "target": target, "depth": depth, "derivables": derivables}

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
        distractor_components = [
            self._get_components(distractor_recipe)
            for distractor_recipe in distractor_recipes
        ]

        if qed:
            facts = recipe_components["facts"] + [
                antecedent
                for distractor in distractor_components
                for antecedent in distractor["rules"][0][0]
            ]
            # Remove any recipe derivables from the facts
            facts = [fact for fact in facts if fact not in recipe_components["derivables"]]
            return {
                "rules": recipe_components["rules"]
                + [distractor["rules"][0] for distractor in distractor_components],
                "facts": facts,
                "target": recipe_components["target"],
                "qed": True,
                "distractor_rules": [distractor["rules"][0] for distractor in distractor_components],
                "depth": recipe_components["depth"],
            }

        # If not qed, choose a random subset of rules and facts (cannot include all facts)
        facts = random.sample(
            recipe_components["facts"],
            random.randint(0, len(recipe_components["facts"]) - 1),
        ) + [
            antecedent
            for distractor in distractor_components
            for antecedent in distractor["rules"][0][0]
        ]
        # Remove any recipe derivables from the facts
        facts = [fact for fact in facts if fact not in recipe_components["derivables"]]
        return {
            "rules": random.sample(
                recipe_components["rules"],
                random.randint(0, len(recipe_components["rules"])),
            )
            + [distractor["rules"][0] for distractor in distractor_components],
            "facts": facts,
            "target": recipe_components["target"],
            "qed": False,
            "distractor_rules": [distractor["rules"][0] for distractor in distractor_components],
            "depth": recipe_components["depth"],
        }
    
    def _get_chain_of_thought_for_recipe_with_antecedents(self, recipe: dict):
        """Get the sequence of derivable facts for a recipe including the antecedents that are used to derive the facts."""

        base_facts = recipe["facts"]
        rules = recipe["rules"]
        derivable_facts = [{"antedecents": [], "fact": fact} for fact in base_facts]
        can_derive = True

        while can_derive:
            new_facts = []
            new_antecedents = []
            for rule in rules:
                if all(antecedent in [fact["fact"] for fact in derivable_facts] for antecedent in rule[0]):
                    new_facts.append(rule[1])
                    new_antecedents.append(rule[0])
            # If new facts are already derivable, stop
            if all(fact in [fact["fact"] for fact in derivable_facts] for fact in new_facts):
                can_derive = False
            for antecedents, fact in zip(new_antecedents, new_facts):
                # If the fact is already derivable, skip
                if fact in [fact["fact"] for fact in derivable_facts]:
                    continue
                derivable_facts.append({"antecedents": antecedents, "fact": fact})

        return derivable_facts[len(base_facts):]

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
                # Cut off the recipe at num_steps
                # print(f"Recipe: {recipe}")
                if self.num_steps >= 0:
                    recipe_with_num_steps = [
                        rule for rule in recipe if rule[2] <= self.num_steps
                    ]
                else:
                    recipe_with_num_steps = recipe
                # Add the recipe to the list of all recipes
                all_recipes.append(recipe_with_num_steps)

        print("Number of recipes: ", len(all_recipes))
        all_recipes = self._remove_redundant_recipes(all_recipes)
        print("Number of unique recipes: ", len(all_recipes))
        all_recipes = self._remove_loops(all_recipes)
        print("Number of unique recipes after removing loops: ", len(all_recipes))

        # for recipe in all_recipes:
        #     print(f"Recipe after unique: {recipe}")

        return all_recipes

    def _generate_dataset_with_distractions(self):
        """Generate the dataset with distractors.
        The dataset is generated by taking the original dataset and adding distractors to it.
        The size of the new dataset matches the requirement by creating multiple copies of the samples from the original dataset and randomly adding distractions.
        The dataset is balanced with half the samples being qed and the other half being non-qed.
        """

        dataset = []
        num_samples_per_recipe = self.dataset_len // len(self.dataset)
        # TODO: Add support for num_samples_per_recipe < 2
        if num_samples_per_recipe < 2:
            raise Exception("Number of samples per recipe should be at least 2.")
        num_qed_samples = num_samples_per_recipe // 2
        for recipe in self.dataset:
            qed_samples = [
                self._get_components_with_distrations(recipe, qed=True)
                for _ in range(num_qed_samples)
            ]
            non_qed_samples = [
                self._get_components_with_distrations(recipe, qed=False)
                for _ in range(num_samples_per_recipe - num_qed_samples)
            ]
            dataset.extend(qed_samples + non_qed_samples)
        return dataset

    def __getitem__(self, idx):
        recipe = self.dataset[idx]
        qed = recipe["qed"]
        if qed:
            labels = torch.tensor(1).long()
        else:
            labels = torch.tensor(0).long()

        cot_states = self._get_chain_of_thought_for_recipe_with_antecedents(recipe)

        if self.example_format == "tags":
            item = stringify_recipe_with_tags(recipe, self.task_type, cot_states)
        elif self.example_format == "text":
            item = stringify_recipe_with_text(recipe, self.task_type, cot_states)
        else:
            raise Exception("Example format not supported. Supported example formats: text, tags")
        if not self.tokenizer:
            return Exception("Tokenizer not provided.")
        encoding = self.tokenizer(item, truncation=True, padding=self.padding)

        return {
            "data": item,
            "labels": labels,
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "recipe": recipe,
            "cot_states": cot_states,
            "num_vars": recipe["num_vars"] if "num_vars" in recipe else None,
            "target": recipe["target"].replace("minecraft:", "").replace("_", " ") if self.example_format == "text"
                    else recipe["target"].replace("minecraft:", "")
        }


class MinecraftAutoregKStepsNVarsDataset(MinecraftAutoregKStepsBaseDataset):
    """Dataset for generating recipes with a specified number of variables."""

    def __init__(
        self,
        num_steps: int,
        num_vars_range: tuple[int, int],
        dataset_len: int,
        max_num_distractors: int = 2,  # Not used
        max_num_distractors_triggerable: int = 99999,
        seed: int = 101,
        tokenizer=None,
        padding: str = "longest",
        task_type: str = "next_token_prediction",    # Binary classification or Next Token Prediction
        example_format: str = "text",                 # Text or Tags
        shuffle: bool = True,
        calculate_num_vars_stats: bool = False
    ):
        assert (
            num_vars_range[0] <= num_vars_range[1]
        ), "Minimum number of variables should be less than or equal to maximum number of variables."
        self.min_num_vars = num_vars_range[0]
        self.max_num_vars = num_vars_range[1]
        self.max_num_distractors_triggerable = max_num_distractors_triggerable
        super().__init__(
            num_steps=num_steps,
            dataset_len=dataset_len,
            max_num_distractors=max_num_distractors,
            seed=seed,
            tokenizer=tokenizer,
            padding=padding,
            task_type=task_type,
            example_format=example_format,
            shuffle=shuffle
        )
        if calculate_num_vars_stats:
            self.num_vars_histogram = [k["num_vars"] for k in self.dataset]
            self.num_vars_histogram = {
                k: self.num_vars_histogram.count(k) for k in self.num_vars_histogram
            }
            self.num_vars_histogram_with_samples = {
                k: [recipe for recipe in self.dataset if recipe["num_vars"] == k]
                for k in self.num_vars_histogram
            }

    def _get_components_with_distrations(self, recipe: list, qed: bool = True):
        """Get the components of a recipe with distractors such that the number of variables is within the specified range.
        Distractors are recipes that can be derived but are not the target.

        Args:
            recipe (list): The recipe
            qed (bool, optional): Whether the recipe is qed or not. Defaults to True.

        Returns:
            components (dict): Dictionary containing the components (rules, facts, target)
        """
        recipe_components = self._get_components(recipe)
        # print("Recipe components: ", recipe_components)
        # Number of vars is the number of unique items in the recipe (antecedents + consequents)
        vars_in_recipe = set(
            [antecedent for rule in recipe for antecedent in rule[0]]
            + [rule[1] for rule in recipe]
        )
        num_vars_in_recipe = len(vars_in_recipe)

        expected_num_vars = random.randint(self.min_num_vars, self.max_num_vars)

        distractor_components = []
        while num_vars_in_recipe < expected_num_vars:
            distractor_recipe = random.choice(self.dataset)
            distractor_vars = set(
                [antecedent for rule in distractor_recipe for antecedent in rule[0]]
                + [rule[1] for rule in distractor_recipe]
            )

            # If the number of variables in the distractor recipe is greater than the maximum number of variables, skip
            try_vars_in_recipe = vars_in_recipe.union(distractor_vars)
            if len(try_vars_in_recipe) > self.max_num_vars:
                break
            # If the distractor recipe adds no new variables, skip
            if len(try_vars_in_recipe) == num_vars_in_recipe:
                continue
            vars_in_recipe = try_vars_in_recipe
            num_vars_in_recipe = len(vars_in_recipe)
            distractor_components.append(self._get_components(distractor_recipe))

        # Limit the number of distractors to the maximum number of distractors triggerable
        # Only consider components where the antecedents are not in recipe derivables
        components_for_facts = [
                distractor for distractor in distractor_components if all(antecedent not in recipe_components["derivables"] for antecedent in distractor["rules"][0][0])
            ]
        # print(f"Recipe components: {recipe_components}")
        # # Print the distractor components that are not considered
        # print(f"Distractor components not considered:")
        # for distractor in distractor_components:
        #     if distractor not in components_for_facts:
        #         print(distractor)
        if len(distractor_components) > self.max_num_distractors_triggerable:
            distractor_facts = set(antecedent for distractor in random.sample(components_for_facts, min(self.max_num_distractors_triggerable, len(components_for_facts))) for antecedent in distractor["rules"][0][0])
        else:
            distractor_facts = set(antecedent for distractor in components_for_facts for antecedent in distractor["rules"][0][0])
        # distractor_facts = random.sample(distractor_facts, min(self.max_num_distractors_triggerable, len(distractor_facts)))

        if qed:
            facts = set(recipe_components["facts"]).union(distractor_facts)
            return {
                "rules": recipe_components["rules"]
                + [distractor["rules"][0] for distractor in distractor_components],
                "facts": list(facts),
                "target": recipe_components["target"],
                "qed": True,
                "num_vars": num_vars_in_recipe,
                "vars_in_recipe": vars_in_recipe,
                "distractor_rules": [distractor["rules"][0] for distractor in distractor_components],
                "depth": recipe_components["depth"]
            }

        # If not qed, choose a random subset of facts (cannot include all facts)
        facts = random.sample(
            recipe_components["facts"],
            random.randint(0, len(recipe_components["facts"]) - 1),
        ) + list(distractor_facts)
        facts = set(facts)
        return {
            "rules": recipe_components["rules"]
            + [distractor["rules"][0] for distractor in distractor_components],
            "facts": list(facts),
            "target": recipe_components["target"],
            "qed": False,
            "num_vars": num_vars_in_recipe,
            "vars_in_recipe": vars_in_recipe,
            "distractor_rules": [distractor["rules"][0] for distractor in distractor_components],
            "depth": recipe_components["depth"]
        }
    
class MinecraftAutoregKStepsNVarsSupressAttackDataset(MinecraftAutoregKStepsNVarsDataset):
    """Dataset for generating recipes with a specified number of variables."""

    def __init__(
        self,
        num_steps: int,
        num_vars_range: tuple[int, int],
        dataset_len: int,
        max_num_distractors: int = 2,  # Not used
        max_num_distractors_triggerable: int = 99999,
        seed: int = 101,
        tokenizer=None,
        padding: str = "longest",
        task_type: str = "next_token_prediction",    # Binary classification or Next Token Prediction
        example_format: str = "text",                 # Text or Tags
        adv_params: dict = None,
        model: torch.nn.Module = None,          # Model to be attacked
        num_samples_per_recipe: int = 2,
        shuffle: bool = True
    ):
        
        self.model = model
        self.adv_params = adv_params
        self.idx_to_recipe = {}
        self.num_samples_per_recipe = num_samples_per_recipe
        self.shuffle_dataset = shuffle
        super().__init__(
            num_steps=num_steps,
            num_vars_range=num_vars_range,
            dataset_len=dataset_len,
            max_num_distractors=max_num_distractors,
            max_num_distractors_triggerable=max_num_distractors_triggerable,
            seed=seed,
            tokenizer=tokenizer,
            padding=padding,
            task_type=task_type,
            example_format=example_format,
            shuffle=False       # Do not shuffle the dataset initially, to retain the idx to recipe mapping
        )

    def _generate_dataset_with_distractions(self):
        """Generate the dataset with distractors.
        The dataset is generated by taking the original dataset and adding distractors to it.
        The size of the new dataset matches the requirement by creating multiple copies of the samples from the original dataset and randomly adding distractions.
        The dataset is balanced with half the samples being qed and the other half being non-qed.
        """

        if self.shuffle_dataset:
            random.seed(self.seed)
            random.shuffle(self.dataset)

        dataset = {}
        idx = 0
        num_samples_per_recipe = self.num_samples_per_recipe
        # TODO: Add support for num_samples_per_recipe < 2
        if num_samples_per_recipe < 2:
            raise Exception("Number of samples per recipe should be at least 2.")
        num_qed_samples = num_samples_per_recipe // 2
        for recipe in self.dataset:
            qed_samples = [
                self._get_components_with_distrations(recipe, qed=True)
                for _ in range(num_qed_samples)
            ]
            non_qed_samples = [
                self._get_components_with_distrations(recipe, qed=False)
                for _ in range(num_samples_per_recipe - num_qed_samples)
            ]
            # Alternate between qed and non-qed samples
            dataset[idx] = []
            for i in range(num_samples_per_recipe):
                if i % 2 == 0:
                    dataset[idx].append(qed_samples[i // 2])
                else:
                    dataset[idx].append(non_qed_samples[i // 2])
            self.idx_to_recipe[idx] = recipe
            idx += 1
        print(f"Found {len(dataset)} recipes.")
        return dataset
    
    def __getitem__(self, idx):
        recipes = self.dataset[idx]
        items = []

        for recipe in recipes:
            # Use getitem from the parent class
            qed = recipe["qed"]
            original_rules = recipe["rules"]
            recipe["rules"] = recipe["distractor_rules"]
            if qed:
                labels = torch.tensor(1).long()
            else:
                labels = torch.tensor(0).long()

            adv_cot_states = self._get_chain_of_thought_for_recipe_with_antecedents(recipe)

            recipe["rules"] = original_rules
            cot_states = self._get_chain_of_thought_for_recipe_with_antecedents(recipe)

            if self.example_format == "tags":
                item = stringify_recipe_with_tags(recipe, self.task_type, cot_states)
                adv_item = stringify_recipe_with_tags(recipe, self.task_type, adv_cot_states)
            elif self.example_format == "text":
                item = stringify_recipe_with_text(recipe, self.task_type, cot_states)
                adv_item = stringify_recipe_with_text(recipe, self.task_type, adv_cot_states)
            else:
                raise Exception("Example format not supported. Supported example formats: text, tags")
            if not self.tokenizer:
                return Exception("Tokenizer not provided.")
            encoding = self.tokenizer(item, truncation=True, padding=self.padding)

            items.append({
                "data": item,
                "adv_data": adv_item,
                "labels": labels,
                "input_ids": encoding.input_ids,
                "attention_mask": encoding.attention_mask,
                "recipe": recipe,
                "cot_states": cot_states,
                "num_vars": recipe["num_vars"] if "num_vars" in recipe else None,
                "target": recipe["target"].replace("minecraft:", "").replace("_", " ") if self.example_format == "text"
                        else recipe["target"].replace("minecraft:", ""),
                "recipe_rules": original_rules,
                "distractor_rules": recipe["distractor_rules"],
                "recipe_from_idx": self.idx_to_recipe[idx],
                "facts": recipe["facts"],
                "depth": recipe["depth"],
            })

        return items

        
