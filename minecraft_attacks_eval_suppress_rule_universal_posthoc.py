import sys
from pathlib import Path

PROJ_ROOT = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, PROJ_ROOT)

from argparse import ArgumentParser
from minecraft.dataset import MinecraftAutoregKStepsNVarsSupressAttackDataset
import json
import itertools
import os
import numpy as np

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from minecraft_attacks_eval_suppress_rule_universal import (
    load_next_state_model_from_wandb,
    minecraftexp_args_to_wandb_run_name,
    get_param_dict_list_from_config_file,
    generation_matches_label,
)
from minecraft_attacks_eval_suppress_rule_universal import (
    get_prompt_and_label,
    generate_text,
    get_rule_suppressed_adv_target,
)
from minecraft_attacks_utils import generation_not_contains_label, generate_text_batch

from transformers import utils

utils.logging.set_verbosity_error()  # Suppress standard warnings

# Suppress the warnings from all libraries
import warnings

warnings.filterwarnings("ignore")

# Suppress the warnings from the transformers library
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# Set the random seed for NumPy
np.random.seed(20)

# Set the random seed for PyTorch
torch.manual_seed(20)

# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(20)
torch._C._jit_set_bailout_depth(0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the config file with model and dataset details",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="minecraft_gcg_results",
        help="Path to the output directory where the results will be dumped",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the experiments on",
    )
    parser.add_argument(
        "--num_eval_examples",
        type=int,
        default=10,
        help="Number of examples to evaluate the attack on",
    )
    parser.add_argument(
        "--num_test_samples_per_rule",
        type=int,
        default=20,
        help="Number of samples to generate for testing",
    )
    parser.add_argument(
        "--only_qed",
        action="store_true",
        help="Only consider the qed value for the attack",
    )
    parser.add_argument(
        "--amnesia_rule_depth",
        type=int,
        default=-1,
        help="Depth of the rules to remove for amnesia (if multiple rules as depth, one with randomly be chosen). Default (-1) is the general suppress rule attack",
    )
    parser.add_argument(
        "--suppression_rule_depth",
        type=int,
        default=-1,
        help="Depth of the rules to start suppressing from. Default (-1) is the general suppress rule attack",
    )

    args = parser.parse_args()
    config_file = args.config_file

    os.makedirs(args.output_dir, exist_ok=True)

    config = get_param_dict_list_from_config_file(args.config_file)
    print("Running with config: ", config)

    for param_dict in config:
        print("Param dict:", param_dict)
        for param_tuple in itertools.product(*param_dict.values()):
            param_instance = {}
            for i, param in enumerate(param_tuple):
                param_instance[list(param_dict.keys())[i]] = param

            param_instance["use_pretrained"] = (
                True if param_instance["model_name"] == "gpt2" else False
            )
            param_instance["max_num_distractors_triggerable"] = param_instance.get(
                "max_num_distractors_triggerable", 3
            )

            exp_id = minecraftexp_args_to_wandb_run_name(param_instance)

            artifact_id = f"model-{exp_id}:v0"

            model = AutoModelForCausalLM.from_pretrained("gpt2")
            model = load_next_state_model_from_wandb(artifact_id, model)

            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.padding_side = "left"

            if args.amnesia_rule_depth >= 0:
                exp_id = exp_id + f"_amnesia_depth{args.amnesia_rule_depth}"
            elif args.suppression_rule_depth > 0:
                exp_id = exp_id + f"_suppression_depth{args.suppression_rule_depth}"

            samples_file = os.path.join(args.output_dir, f"{exp_id}.json")
            print(f"Loading samples from {samples_file}")
            # Load samples from the file
            with open(samples_file, "r") as f:
                samples = json.load(f)

            model.to(args.device)
            model.eval()
            print(
                f"Length of model embeddings: {model.get_input_embeddings().weight.shape[0]}"
            )
            model.resize_token_embeddings(len(tokenizer))
            print(
                f"Length of model embeddings after resizing: {model.get_input_embeddings().weight.shape[0]}"
            )
            print(f"Model vocab size: {model.config.vocab_size}")

            eval_dataset = MinecraftAutoregKStepsNVarsSupressAttackDataset(
                num_steps=param_instance["num_steps"],
                num_vars_range=(
                    param_instance["min_num_vars"],
                    param_instance["max_num_vars"],
                ),
                max_num_distractors_triggerable=param_instance.get(
                    "max_num_distractors_triggerable", 3
                ),
                example_format=param_instance.get("example_format", "text"),
                dataset_len=param_instance.get("eval_len"),
                # dataset_len=max(param_instance.get("eval_len"), args.num_eval_examples),
                seed=param_instance["seed"],
                tokenizer=tokenizer,
                task_type="next_token_prediction",
                num_samples_per_recipe=args.num_test_samples_per_rule,
                only_qed=args.only_qed,
            )

            percentage_successful_suppressions = []
            percentage_successful_attacks = []

            for sample in tqdm(samples, desc="Running attack on the eval dataset"):
                sample["final_adversarial_suffix"] = sample["adversarial_suffix"]
                recipe_suppressed = sample["recipe_suppressed"]
                if "original_recipe" in sample:
                    recipe_suppressed = sample["original_recipe"]
                # Sort sample first by depth and then by the consequent
                rule_supressed_in_sample = sorted(
                    recipe_suppressed, key=lambda x: (x[2], x[1])
                )
                # Sort antecedents in each rule
                rule_supressed_in_sample = tuple(
                    (tuple(sorted(antecedents)), consequent, depth)
                    for (antecedents, consequent, depth) in rule_supressed_in_sample
                )
                # print(rule_supressed_in_sample)

                if rule_supressed_in_sample not in eval_dataset.recipe_to_idx:
                    print("Couldn't find a matching recipe in the eval dataset")
                    continue

                i = eval_dataset.recipe_to_idx[rule_supressed_in_sample]
                print("Found a matching recipe!")

                recipes = eval_dataset[i]
                num_successful_suppressions = 0
                num_successful_attacks = 0

                recipe_data = [recipe["data"] for recipe in recipes]
                prompts, labels = zip(*map(get_prompt_and_label, recipe_data))
                generations = generate_text_batch(
                    model,
                    tokenizer,
                    prompts,
                    device=args.device,
                    batch_size=len(prompts),
                )

                recipe_from_idx = sample["recipe_suppressed"]

                adv_tgts, recipes_suppressed, suppressed_strings = zip(
                    *map(
                        get_rule_suppressed_adv_target,
                        generations,
                        [recipe_from_idx] * len(generations),
                        [True] * len(generations),
                    )
                )

                suppressed_strings_in_generation = map(
                    generation_not_contains_label, suppressed_strings, generations
                )
                suppressed_strings_in_generation = [
                    not s for s in suppressed_strings_in_generation
                ]

                generations = [
                    generation.split("I cannot create any other items.")[0]
                    for generation in generations
                ]

                attack_prompts = [
                    f"{prompt}{sample['final_adversarial_suffix']}"
                    for prompt in prompts
                ]
                attack_generations = generate_text_batch(
                    model,
                    tokenizer,
                    attack_prompts,
                    device=args.device,
                    batch_size=len(attack_prompts),
                )

                attack_generation_matches_adv_target = list(
                    map(generation_matches_label, adv_tgts, attack_generations)
                )
                suppressions_successful = list(
                    map(
                        generation_not_contains_label,
                        suppressed_strings,
                        attack_generations,
                    )
                )

                num_suppressions_in_original = sum(suppressed_strings_in_generation)
                num_successful_suppressions = sum(suppressions_successful)
                num_successful_attacks = sum(attack_generation_matches_adv_target)

                print(
                    f"Train attack success percentage: {sample['attack_success_percentage']}"
                )
                print(
                    f"Number of suppressed strings in original generations: {num_suppressions_in_original}/{len(recipes)}"
                )
                print(
                    f"Number of successful suppressions: {num_successful_suppressions}/{len(recipes)}"
                )
                print(
                    f"Number of successful attacks: {num_successful_attacks}/{len(recipes)}"
                )

                sample["num_other_examples"] = len(recipes)
                sample["num_suppressions_in_original"] = num_suppressions_in_original
                sample["num_successful_suppressions"] = num_successful_suppressions
                sample["num_successful_attacks"] = num_successful_attacks
                sample[
                    "percentage_suppressions_in_original"
                ] = num_suppressions_in_original / len(recipes)
                sample[
                    "percentage_successful_suppressions"
                ] = num_successful_suppressions / len(recipes)
                sample["percentage_successful_attacks"] = num_successful_attacks / len(
                    recipes
                )

                sample["other_examples"] = [
                    {
                        "attack_prompt": ap,
                        "label": l,
                        "original_generation": og,
                        "adv_target": at,
                        "suppressed_string": ss,
                        "attack_generation": ag,
                        "suppression_successful": ssuc,
                        "attack_generation_matches_adv_target": agmat,
                        "suppressed_string_in_generation": ssig,
                    }
                    for ap, l, og, at, ss, ag, ssuc, agmat, ssig in zip(
                        attack_prompts,
                        labels,
                        generations,
                        adv_tgts,
                        suppressed_strings,
                        attack_generations,
                        suppressions_successful,
                        attack_generation_matches_adv_target,
                        suppressed_strings_in_generation,
                    )
                ]
                del sample["train_sample_results"]
                percentage_successful_suppressions.append(
                    num_successful_suppressions / len(recipes)
                )
                percentage_successful_attacks.append(
                    num_successful_attacks / len(recipes)
                )

            # Save the samples with the results
            with open(
                os.path.join(
                    args.output_dir, f"suppressed_results_batched_{exp_id}.json"
                ),
                "w",
            ) as f:
                json.dump(samples, f)

            print(
                f"Mean suppression percentage: {sum(percentage_successful_suppressions)/len(percentage_successful_suppressions)}"
            )
            print(
                f"Max suppression percentage: {max(percentage_successful_suppressions)}"
            )
            print(
                f"Min suppression percentage: {min(percentage_successful_suppressions)}"
            )

            print(
                f"Mean attack success percentage: {sum(percentage_successful_attacks)/len(percentage_successful_attacks)}"
            )
            print(
                f"Max attack success percentage: {max(percentage_successful_attacks)}"
            )
            print(
                f"Min attack success percentage: {min(percentage_successful_attacks)}"
            )
