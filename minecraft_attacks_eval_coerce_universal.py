import sys
from pathlib import Path

PROJ_ROOT = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, PROJ_ROOT)

from argparse import ArgumentParser
from minecraft.dataset import MinecraftAutoregKStepsNVarsCoerceAttackDataset
import json
import itertools
import os
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import trange

from llm_attacks import get_nonascii_toks
from minecraft_attacks_utils import *

from transformers import utils

utils.logging.set_verbosity_error()  # Suppress standard warnings

# Suppress the warnings from all libraries
import warnings

warnings.filterwarnings("ignore")

# Suppress the warnings from the transformers library
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

import time

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
        "--adv_init",
        type=str,
        default="I have I have I have I have I I I I I have",
        help="The initial adversarial string to start the attack with",
    )
    parser.add_argument(
        "--adv_target",
        type=str,
        default="I have nothing and so I can create everything.",
        help="The target adversarial string to achieve",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of GCG optimization steps to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the GCG optimization",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=256,
        help="Top k tokens to consider for the GCG optimization",
    )
    parser.add_argument(
        "--allow_non_ascii",
        action="store_true",
        help="Allow non-ASCII characters in the generated strings",
    )
    parser.add_argument(
        "--allow_tokens_not_in_dataset",
        action="store_true",
        help="Allow tokens that are not in the dataset",
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
        "--num_train_samples_per_rule",
        type=int,
        default=10,
        help="Number of samples to generate per rule",
    )
    parser.add_argument(
        "--num_test_samples_per_rule",
        type=int,
        default=10,
        help="Number of samples to generate for testing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to run the attack for",
    )
    parser.add_argument(
        "--only_qed",
        action="store_true",
        help="Only consider the qed value for the attack",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload existing results from directory",
    )

    args = parser.parse_args()
    total_num_steps = args.num_steps
    original_adv_init = args.adv_init
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

            eval_dataset = MinecraftAutoregKStepsNVarsCoerceAttackDataset(
                num_steps=param_instance["num_steps"],
                num_vars_range=(
                    param_instance["min_num_vars"],
                    param_instance["max_num_vars"],
                ),
                max_num_distractors_triggerable=param_instance.get(
                    "max_num_distractors_triggerable", 3
                ),
                example_format=param_instance.get("example_format", "text"),
                dataset_len=max(param_instance.get("eval_len"), args.num_eval_examples),
                seed=param_instance["seed"],
                tokenizer=tokenizer,
                task_type="next_token_prediction",
                num_samples_per_target=args.num_train_samples_per_rule,
                only_qed=args.only_qed,
            )

            not_allowed_tokens = None
            if not args.allow_tokens_not_in_dataset:
                all_tokens_in_vocab = set(range(tokenizer.vocab_size))
                tokens_in_dataset = get_allowed_tokens(tokenizer, eval_dataset)
                not_allowed_tokens = torch.tensor(
                    list(all_tokens_in_vocab - tokens_in_dataset), device=args.device
                )
                print(f"Number of tokens in dataset: {len(tokens_in_dataset)}")
            elif not args.allow_non_ascii:
                not_allowed_tokens = get_nonascii_toks(tokenizer)
                print(f"Number of non-ascii tokens: {len(not_allowed_tokens)}")

            results = []
            targets_to_ignore = []
            if args.reload:
                reload_path = os.path.join(args.output_dir, f"{exp_id}.json")
                print(f"Loading existing results from {reload_path}")
                data = None
                try:
                    data = json.load(open(reload_path))
                except Exception as e:
                    print(
                        f"Coudn't load from file {reload_path} with Exception: {e}. Starting from scratch."
                    )
                if data and len(data) > 0:
                    results = data
                    targets_to_ignore = [r["attack_target"] for r in results]
                    print(
                        f"Found {len(targets_to_ignore)} targets in existing results..."
                    )

            example_idxs_to_ignore = [
                eval_dataset.target_to_idx[r] for r in targets_to_ignore
            ]
            samples, prompts = get_samples_from_dataset(
                eval_dataset,
                args.num_eval_examples - len(targets_to_ignore),
                param_instance["num_steps"] - 1,
                model,
                tokenizer,
                device=args.device,
                example_idxs_to_ignore=example_idxs_to_ignore,
            )
            start_idx = len(results)

            for epoch in range(args.num_epochs):
                num_successes = 0
                acc = 0
                # Maximum number of retries if CUDA runs out of memory
                max_num_tries = 5

                # print(f"Running attack for {exp_id} | Start idx: {start_idx} | Num total: {num_total} | Accuracy: {acc}")
                ted_bar = trange(
                    start_idx,
                    start_idx + len(samples),
                    desc=f"Epoch: {epoch} | Running accuracy: {acc}",
                )
                for eval_i in ted_bar:
                    recipes = samples[eval_i - start_idx]
                    num_tries = 0
                    # args.num_steps = total_num_steps
                    args.adv_init = original_adv_init

                    # Get adv_init from last epoch's results
                    if len(results) > 0:
                        last_epoch_results = [
                            r
                            for r in results
                            if r["eval_idx"] == eval_i and r["epoch"] == epoch - 1
                        ]
                        if len(last_epoch_results) > 0:
                            args.adv_init = last_epoch_results[0]["adversarial_suffix"]

                    epoch_adv_init = args.adv_init

                    ted_bar.set_description(
                        f"Epoch: {epoch} | Running accuracy: {acc} | Example Idx: {eval_i}",
                        refresh=True,
                    )

                    idx_results = []

                    for recipe in recipes:
                        prompt, label = get_prompt_and_label(recipe["data"])
                        generation = generate_text(
                            model, tokenizer, prompt, device=args.device
                        )
                        args.adv_target = recipe["attack_target"].strip()
                        args.adv_target += "\nI cannot create any other items."
                        # args.adv_target += "\nI cannot create any other items."

                        generation = generation.split(
                            "I cannot create any other items."
                        )[0]

                        print(f"Generation: {generation}")
                        print(f"Adv init: {args.adv_init}")
                        print(f"Adv target: {args.adv_target}")

                        attack_result = {}
                        success = False
                        num_tries = 0
                        attack_start_time = time.time()

                        while num_tries < max_num_tries:
                            try:
                                success, attack_result = run_gcg_attack(
                                    args,
                                    model,
                                    tokenizer,
                                    prompt,
                                    label,
                                    not_allowed_tokens,
                                )
                                break
                            except RuntimeError as e:
                                log_final_metrics(
                                    args.output_dir,
                                    exp_id,
                                    num_successes,
                                    len(results),
                                    acc,
                                )
                                print(
                                    f"Caught exception: {e} | Trying again in 5 seconds..."
                                )
                                torch.cuda.empty_cache()
                                time.sleep(25)
                                num_tries += 1
                                continue

                        if "cuda" in args.device:
                            torch.cuda.synchronize(device=args.device)
                        attack_end_time = time.time()

                        attack_result["original_generation"] = generation
                        attack_result[
                            "original_generation_matches_label"
                        ] = generation_matches_label(
                            label, attack_result["original_generation"]
                        )
                        attack_result["attack_target"] = args.adv_target
                        attack_result["target_in_original_generation"] = (
                            args.adv_target == attack_result["original_generation"]
                        )
                        attack_result["attack_time"] = (
                            attack_end_time - attack_start_time
                        )

                        attack_result[
                            "attack_prompt"
                        ] = f"{prompt}{attack_result['adversarial_suffix']}"
                        attack_result["attack_generation"] = generate_text(
                            model,
                            tokenizer,
                            attack_result["attack_prompt"],
                            device=args.device,
                        )
                        attack_result[
                            "attack_generation_matches_label"
                        ] = generation_matches_label(
                            label, attack_result["attack_generation"]
                        )
                        attack_result["attack_success"] = success
                        attack_result["qed"] = recipe["labels"].item()
                        idx_results.append(attack_result)

                        # update adv_init if the attack was successful
                        if success:
                            args.adv_init = attack_result["adversarial_suffix"]

                    if sum([r["attack_success"] for r in idx_results]) == len(
                        idx_results
                    ):
                        num_successes += 1

                    acc = num_successes / (eval_i + 1)

                    final_idx_results = {
                        "eval_idx": eval_i,
                        "epoch": epoch,
                        "adv_init": epoch_adv_init,
                        "adversarial_suffix": args.adv_init,
                        "attack_success": success and len(idx_results) == len(recipes),
                        "attack_success_percentage": sum(
                            [r["attack_success"] for r in idx_results]
                        )
                        / len(idx_results),
                        # "recipe_suppressed": [(list(antecedents), consequent, depth) for (antecedents, consequent, depth) in recipe["recipe_from_idx"]],
                        # "num_steps": total_num_steps - args.num_steps,
                        "num_steps": sum([r["num_steps"] for r in idx_results]),
                        "total_attack_time": sum(
                            [r["attack_time"] for r in idx_results]
                        ),
                        "train_sample_results": idx_results,
                        "train_running_accuracy": acc,
                    }

                    results.append(final_idx_results)

                    # Update the json file with the attack result
                    with open(
                        os.path.join(args.output_dir, f"{exp_id}.json"), "w"
                    ) as f:
                        json.dump(results, f)

            log_final_metrics(args.output_dir, exp_id, num_successes, len(results), acc)
