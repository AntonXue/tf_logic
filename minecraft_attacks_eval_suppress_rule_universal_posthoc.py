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
import torch.nn as nn
from tqdm import tqdm
from minecraft_attacks_eval_suppress_rule_universal import load_next_state_model_from_wandb, minecraftexp_args_to_wandb_run_name, get_param_dict_list_from_config_file, generation_matches_label
from minecraft_attacks_eval_suppress_rule_universal import get_prompt_and_label, generate_text, get_rule_suppressed_adv_target

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


def generation_not_contains_label(label, generation):
    all_targets_in_labels = [t.split(".")[0].strip() for t in label.split("so I can create ")[1:]]
    all_targets_in_labels = set(["so I can create " + target for target in all_targets_in_labels])
    all_targets_in_generation = [t.split(".")[0].strip() for t in generation.split("so I can create ")[1:]]
    all_targets_in_generation = set(["so I can create " + target for target in all_targets_in_generation])

    # Check if no target in the label is present in the generation
    return len(all_targets_in_labels.intersection(all_targets_in_generation)) == 0

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

            param_instance["use_pretrained"] = True if param_instance["model_name"] == "gpt2" else False
            param_instance["max_num_distractors_triggerable"] = param_instance.get("max_num_distractors_triggerable", 3)

            exp_id = minecraftexp_args_to_wandb_run_name(param_instance)

            samples_file = os.path.join(args.output_dir, f"{exp_id}.json")
            print(f"Loading samples from {samples_file}")
            # Load samples from the file
            with open(samples_file, "r") as f:
                samples = json.load(f)

            artifact_id = f"model-{exp_id}:v0"

            model = AutoModelForCausalLM.from_pretrained("gpt2")
            model = load_next_state_model_from_wandb(artifact_id, model)

            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

            model.to(args.device)
            model.eval()
            print(f"Length of model embeddings: {model.get_input_embeddings().weight.shape[0]}")
            model.resize_token_embeddings(len(tokenizer))
            print(f"Length of model embeddings after resizing: {model.get_input_embeddings().weight.shape[0]}")
            print(f"Model vocab size: {model.config.vocab_size}")

            eval_dataset = MinecraftAutoregKStepsNVarsSupressAttackDataset(
                                num_steps=param_instance["num_steps"],
                                num_vars_range=(param_instance["min_num_vars"], param_instance["max_num_vars"]),
                                max_num_distractors_triggerable=param_instance.get("max_num_distractors_triggerable", 3),
                                example_format=param_instance.get("example_format", "text"),
                                dataset_len=param_instance.get("eval_len"),
                                # dataset_len=max(param_instance.get("eval_len"), args.num_eval_examples),
                                seed=param_instance["seed"],
                                tokenizer=tokenizer,
                                task_type="next_token_prediction",
                                num_samples_per_recipe=args.num_test_samples_per_rule,
                                only_qed=args.only_qed
                            )

            # Find examples from the eval_dataset for each sample in the samples
            for sample in tqdm(samples, desc="Finding examples in the eval dataset"):
                # Let the final adversarial suffix for the sample be the suffix for the last sample_result where attack_success is True
                sample["final_adversarial_suffix"] = sample["adversarial_suffix"]
                # Sort sample first by depth and then by the consequent
                rule_supressed_in_sample = sorted(sample["recipe_suppressed"], key=lambda x: (x[2], x[1]))
                # Sort antecedents in each rule
                rule_supressed_in_sample = [[sorted(antecedents), consequent, depth] for (antecedents, consequent, depth) in rule_supressed_in_sample]
                sample["final_adversarial_suffix"] = sample["adversarial_suffix"]
                print(f"Rule suppressed in sample: {rule_supressed_in_sample}")
                for i in range(len(eval_dataset)):
                    example = eval_dataset[i]
                    recipe = example[0]
                    if recipe['depth'] < param_instance["num_steps"] - 1:
                        continue
                    rule_supressed_in_recipe = [[list(antecedents), consequent, depth] for (antecedents, consequent, depth) in recipe["recipe_from_idx"]]
                    # sort the rules in the recipe and the rules in the sample
                    rule_supressed_in_recipe = sorted(rule_supressed_in_recipe, key=lambda x: (x[2], x[1]))
                    rule_supressed_in_recipe = [[sorted(antecedents), consequent, depth] for (antecedents, consequent, depth) in rule_supressed_in_recipe]
                    if rule_supressed_in_recipe == rule_supressed_in_sample:
                        print("Found the example in the eval dataset")
                        print(f"Recipe in eval dataset: {rule_supressed_in_recipe}")
                        print(f"Recipe in sample: {rule_supressed_in_sample}")
                        sample["other_examples"] = example
                        break

            # Try the adversarial attack on the eval_dataset
            for sample in tqdm(samples, desc="Running attack on the eval dataset"):
                if "other_examples" not in sample:
                    continue
                recipes = sample["other_examples"]
                num_successful_suppressions = 0
                num_successful_attacks = 0

                for k, recipe in enumerate(recipes):
                    prompt, label = get_prompt_and_label(recipe['data'])
                    generation = generate_text(model, tokenizer, prompt, device=args.device)
                    adv_target, recipe_suppressed, suppressed_string = get_rule_suppressed_adv_target(generation, recipe['recipe_from_idx'], return_suppressed_string=True)
                    generation = generation.split("I cannot create any other items.")[0]

                    attack_prompt = f"{prompt} {sample['final_adversarial_suffix']}"
                    attack_generation = generate_text(model, tokenizer, attack_prompt, device=args.device)
                    
                    print(f"Generation: {generation}")
                    print(f"Suppressed recipe: {recipe_suppressed}")
                    print(f"Adv target: {adv_target}")
                    print(f"Attack generation: {attack_generation}")

                    suppressed_string_in_generation = generation_not_contains_label(suppressed_string, generation)
                    print(f"Suppressed string in original generation: {not suppressed_string_in_generation}")

                    attack_generation_matches_adv_target = generation_matches_label(adv_target, attack_generation)
                    print(f"Attack generation matches adv target: {attack_generation_matches_adv_target}")

                    # Check if the suppression was successful
                    suppression_successful = generation_not_contains_label(suppressed_string, attack_generation)
                    print(f"Suppression successful: {suppression_successful}")
                    num_successful_suppressions += suppression_successful
                    num_successful_attacks += attack_generation_matches_adv_target

                    sample["other_examples"][k] = {}

                    sample["other_examples"][k]["attack_prompt"] = attack_prompt
                    sample["other_examples"][k]["label"] = label
                    sample["other_examples"][k]["original_generation"] = generation[len(prompt):].split("I cannot create any other items.")[0]
                    sample["other_examples"][k]["suppressed_string"] = suppressed_string
                    sample["other_examples"][k]["adv_target"] = adv_target
                    sample["other_examples"][k]["attack_generation"] = attack_generation
                    sample["other_examples"][k]["suppression_successful"] = suppression_successful
                    sample["other_examples"][k]["attack_generation_matches_adv_target"] = attack_generation_matches_adv_target
                    sample["other_examples"][k]["suppressed_string_in_generation"] = not suppressed_string_in_generation

                    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

                print(f"Number of successful suppressions: {num_successful_suppressions}/{len(recipes)}")
                print(f"Number of successful attacks: {num_successful_attacks}/{len(recipes)}")
                sample["num_other_examples"] = len(recipes)
                sample["num_successful_suppressions"] = num_successful_suppressions
                sample["num_successful_attacks"] = num_successful_attacks
                sample["percentage_successful_suppressions"] = num_successful_suppressions / len(recipes)
                sample["percentage_successful_attacks"] = num_successful_attacks / len(recipes)

                print("\n============================================================\n")

            # Save the samples with the results
            with open(os.path.join(args.output_dir, f"suppressed_results_{exp_id}.json"), "w") as f:
                    json.dump(samples, f)