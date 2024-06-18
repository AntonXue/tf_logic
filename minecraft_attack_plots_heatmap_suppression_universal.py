import sys
from pathlib import Path

PROJ_ROOT = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, PROJ_ROOT)

from argparse import ArgumentParser
import json
import itertools
import os
import numpy as np

from safetensors import safe_open
import wandb
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from minecraft_attacks_utils import (
    load_next_state_model_from_wandb,
    minecraftexp_args_to_wandb_run_name,
    get_param_dict_list_from_config_file,
)

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

import torch

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

            model.to(args.device)
            model.eval()

            num_tokens_to_generate = 40

            attack_stats = {k: [] for k in range(12)}

            orig_attack_stats = {k: [] for k in range(12)}

            attack_stats_head = {k: {l: [] for l in range(12)} for k in range(12)}

            orig_attack_stats_head = {k: {l: [] for l in range(12)} for k in range(12)}

            string_gray_dye_stats = {
                "string_in_prompt_not_in_suffix": [],
                "string_in_prompt_in_suffix": [],
                "string_not_in_prompt_in_suffix": [],
                "string_not_in_prompt_not_in_suffix": [],
            }

            # samples_file = os.path.join(args.output_dir, f"{exp_id}_True.json")
            samples_file = os.path.join(args.output_dir, f"{exp_id}.json")
            print(f"Loading samples from {samples_file}")
            # Load samples from the file
            with open(samples_file, "r") as f:
                samples = json.load(f)

            for sample_i, example in tqdm(enumerate(samples)):
                found_sample = False
                for j, sample in enumerate(example["train_sample_results"][-1::-1]):
                    if sample["attack_success"] and sample["qed"]:
                        found_sample = True
                        break
                if not found_sample:
                    continue
                input_to_model = sample["original_prompt"]
                attack_input_to_model = sample["attack_prompt"]

                tokenized_input = tokenizer(
                    input_to_model, return_tensors="pt"
                ).input_ids.to(args.device)
                tokenized_attack_input = tokenizer(
                    attack_input_to_model, return_tensors="pt"
                ).input_ids.to(args.device)

                original_tokenized_input = tokenized_input.clone()
                original_tokenized_attack_input = tokenized_attack_input.clone()

                # Analyze attentions of all layers with and without the adversarial suffix
                with torch.no_grad():
                    model.config.output_attentions = True

                    for ntk in range(num_tokens_to_generate):
                        op_without_suffix = model(
                            tokenized_input, output_attentions=True
                        )
                        op_with_suffix = model(
                            tokenized_attack_input, output_attentions=True
                        )

                        # Get list of tokens for input and attack input
                        tokens_without_suffix = tokenizer.convert_ids_to_tokens(
                            tokenized_input[0]
                        )
                        tokens_with_suffix = tokenizer.convert_ids_to_tokens(
                            tokenized_attack_input[0]
                        )

                        # Get the next token
                        next_token_without_suffix = op_without_suffix.logits[
                            0, -1
                        ].argmax()
                        next_token_with_suffix = op_with_suffix.logits[0, -1].argmax()

                        attns_without_suffix = op_without_suffix.attentions
                        attns_with_suffix = op_with_suffix.attentions
                        model.config.output_attentions = False

                        orig_attns_without_suffix = attns_without_suffix
                        orig_attns_with_suffix = attns_with_suffix

                        # Create a heatmap of the attention scores for the last token
                        # Sum the attention scores for all heads in a layer
                        # The heatmap's x-axis is the layer number
                        # The heatmap's y-xis is the token value

                        # Get the attentions for the last token
                        attns_without_suffix = [
                            attns_without_suffix[layer].squeeze(0)
                            for layer in range(len(attns_without_suffix))
                        ]
                        attns_with_suffix = [
                            attns_with_suffix[layer].squeeze(0)
                            for layer in range(len(attns_with_suffix))
                        ]

                        attns_without_suffix = [
                            attns_without_suffix_layer.sum(dim=0)
                            for attns_without_suffix_layer in attns_without_suffix
                        ]
                        attns_with_suffix = [
                            attns_with_suffix_layer.sum(dim=0)
                            for attns_with_suffix_layer in attns_with_suffix
                        ]

                        # Get the attentions for the last token
                        attns_without_suffix = [
                            attns_without_suffix_layer[-1]
                            for attns_without_suffix_layer in attns_without_suffix
                        ]
                        attns_with_suffix = [
                            attns_with_suffix_layer[-1]
                            for attns_with_suffix_layer in attns_with_suffix
                        ]

                        print(
                            f"Shape of attns_without_suffix: {attns_without_suffix[0].shape}"
                        )
                        print(
                            f"Shape of attns_with_suffix: {attns_with_suffix[0].shape}"
                        )

                        # Generate the heatmap

                        heatmap_without_suffix = (
                            torch.stack(attns_without_suffix).cpu().numpy()
                        )
                        heatmap_with_suffix = (
                            torch.stack(attns_with_suffix).cpu().numpy()
                        )

                        # Switch the axes

                        heatmap_without_suffix = heatmap_without_suffix.transpose(1, 0)
                        heatmap_with_suffix = heatmap_with_suffix.transpose(1, 0)

                        print(
                            f"Shape of heatmap_without_suffix: {heatmap_without_suffix.shape}"
                        )
                        print(
                            f"Shape of heatmap_with_suffix: {heatmap_with_suffix.shape}"
                        )

                        # exit()

                        fig, axs = plt.subplots(1, 2, figsize=(15, 50))

                        axs[0].set_title("Attention Heatmap without Suffix")
                        axs[1].set_title("Attention Heatmap with Suffix")

                        # Plot the heatmaps
                        axs[0].imshow(
                            heatmap_without_suffix,
                            cmap="RdYlGn",
                            interpolation="nearest",
                        )
                        axs[1].imshow(
                            heatmap_with_suffix, cmap="RdYlGn", interpolation="nearest"
                        )

                        # Label the axes
                        axs[0].set_xlabel("Layer")
                        axs[0].set_ylabel("Token")

                        # Use tokens as the y-axis
                        axs[0].set_yticks(range(len(tokens_without_suffix)))
                        axs[0].set_yticklabels(tokens_without_suffix)
                        axs[0].set_xticks(range(12))
                        axs[0].set_xticklabels(range(12))

                        axs[1].set_xlabel("Layer")
                        axs[1].set_ylabel("Token")

                        # Use tokens as the y-axis
                        axs[1].set_yticks(range(len(tokens_with_suffix)))
                        axs[1].set_yticklabels(tokens_with_suffix)
                        axs[1].set_xticks(range(12))
                        axs[1].set_xticklabels(range(12))

                        # Add a colorbar
                        fig.colorbar(
                            axs[0].imshow(
                                heatmap_without_suffix,
                                cmap="RdYlGn",
                                interpolation="nearest",
                            ),
                            ax=axs[0],
                        )
                        fig.colorbar(
                            axs[1].imshow(
                                heatmap_with_suffix,
                                cmap="RdYlGn",
                                interpolation="nearest",
                            ),
                            ax=axs[1],
                        )

                        os.makedirs(
                            f"attention_heatmaps_supress_universal/{exp_id}/sample_{sample_i}_{j}",
                            exist_ok=True,
                        )
                        plt.savefig(
                            f"attention_heatmaps_supress_universal/{exp_id}/sample_{sample_i}_{j}/token_{ntk}.png"
                        )

                        for layer in range(len(attns_without_suffix)):
                            attns_without_suffix_layer = orig_attns_without_suffix[
                                layer
                            ].squeeze(0)
                            attns_with_suffix_layer = orig_attns_with_suffix[
                                layer
                            ].squeeze(0)

                            attns_without_suffix_layer_avg = (
                                attns_without_suffix_layer.mean(dim=0)
                            )
                            attns_with_suffix_layer_avg = attns_with_suffix_layer.mean(
                                dim=0
                            )

                            attack_volume = (
                                tokenized_attack_input.shape[-1]
                                - original_tokenized_input.shape[-1]
                            ) / tokenized_attack_input.shape[-1]
                            attack_leverage = (
                                attns_with_suffix_layer_avg[-1][
                                    original_tokenized_input.shape[-1] :
                                ].sum()
                                / attns_with_suffix_layer_avg[-1].sum()
                            )
                            attack_leverage = attack_leverage.item()

                            # print(f"Number of adversarial tokens: {tokenized_attack_input.shape[-1] - original_tokenized_input.shape[-1]}")
                            # print(f"Number of input tokens: {tokenized_input.shape[-1]}")
                            # print(f"Number of adversarial tokens from the attentions: {attns_with_suffix_layer_avg[-1][original_tokenized_input.shape[-1]:].shape}")
                            # print(f"Number of input tokens from the attentions: {attns_with_suffix_layer_avg[-1].shape}")

                            # print(f"(Attack Volume, Attack Leverage) = ({attack_volume},{attack_leverage})")

                            attack_stats[layer].append(
                                (sample_i, ntk, attack_volume, attack_leverage)
                            )

                            for head in range(attns_without_suffix_layer.shape[0]):
                                attns_without_suffix_head = attns_without_suffix_layer[
                                    head
                                ]
                                attns_with_suffix_head = attns_with_suffix_layer[head]

                                attack_leverage = (
                                    attns_with_suffix_head[-1][
                                        original_tokenized_input.shape[-1] :
                                    ].sum()
                                    / attns_with_suffix_head[-1].sum()
                                )
                                attack_leverage = attack_leverage.item()

                                attack_stats_head[layer][head].append(
                                    (sample_i, ntk, attack_volume, attack_leverage)
                                )

                            print("\n--------------------\n")

                        # Update tokenized_input and tokenized_attack_input with the new tokens
                        tokenized_input = torch.cat(
                            [
                                tokenized_input,
                                next_token_without_suffix.unsqueeze(0).unsqueeze(0),
                            ],
                            dim=-1,
                        ).to(args.device)
                        tokenized_attack_input = torch.cat(
                            [
                                tokenized_attack_input,
                                next_token_with_suffix.unsqueeze(0).unsqueeze(0),
                            ],
                            dim=-1,
                        ).to(args.device)

                print(attack_stats)

                # for k in string_gray_dye_stats:
                #     print(f"{k}: {len(string_gray_dye_stats[k])}")
                #     print(f"{string_gray_dye_stats[k]}")
                #     print("--------------------")

                # n_samples -= 1

                # if n_samples == 0:
                #     break

            # Plot the attack stats such that each number of tokens is plotted as a separate color
            # Each plot has (attack volume, attack leverage) as points on the graph

            # Draw a figure for 12 layers
            fig, axs = plt.subplots(4, 3, figsize=(20, 20))

            for layer in range(12):
                if len(attack_stats[layer]) == 0:
                    continue

                attack_stats_layer = attack_stats[layer]
                attack_stats_layer = np.array(attack_stats_layer)

                axs[layer // 3, layer % 3].set_title(f"Layer: {layer}")
                # axs[layer // 3, layer % 3].scatter(attack_stats_layer[:, 1], attack_stats_layer[:, 2])
                # Each sample_i is a different color
                for sample_i in range(len(samples)):
                    sample_i_indices = np.where(attack_stats_layer[:, 0] == sample_i)
                    sample_i_data = attack_stats_layer[sample_i_indices]
                    axs[layer // 3, layer % 3].scatter(
                        sample_i_data[:, 2],
                        sample_i_data[:, 3],
                        label=f"Sample {sample_i}",
                    )

                # Also plot the y = x line
                axs[layer // 3, layer % 3].plot(
                    [0, 1], [0, 1], color="red", linestyle="--"
                )

            # Add a legend
            axs[0, 0].legend()

            plt.savefig(
                f"attention_heatmaps_supress_universal/attack_stats_{exp_id}.png"
            )

        for layer in range(12):
            if len(attack_stats_head[layer][0]) == 0:
                continue

            fig, axs = plt.subplots(4, 3, figsize=(20, 20))

            for head in range(12):
                if len(attack_stats_head[layer][head]) == 0:
                    continue

                attack_stats_head_layer_head = attack_stats_head[layer][head]
                attack_stats_head_layer_head = np.array(attack_stats_head_layer_head)

                axs[head // 3, head % 3].set_title(f"Layer: {layer}, Head: {head}")
                # axs[head // 3, head % 3].scatter(attack_stats_head_layer_head[:, 1], attack_stats_head_layer_head[:, 2])
                # Each sample_i is a different color
                for sample_i in range(len(samples)):
                    sample_i_indices = np.where(
                        attack_stats_head_layer_head[:, 0] == sample_i
                    )
                    sample_i_data = attack_stats_head_layer_head[sample_i_indices]
                    axs[head // 3, head % 3].scatter(
                        sample_i_data[:, 2],
                        sample_i_data[:, 3],
                        label=f"Sample {sample_i}",
                    )

                # Also plot the y = x line
                axs[head // 3, head % 3].plot(
                    [0, 1], [0, 1], color="red", linestyle="--"
                )

            plt.savefig(
                f"attention_heatmaps_supress_universal/attack_stats_{exp_id}_layer_{layer}.png"
            )
