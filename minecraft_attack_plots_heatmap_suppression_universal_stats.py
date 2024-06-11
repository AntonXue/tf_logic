import sys
from pathlib import Path
PROJ_ROOT = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, PROJ_ROOT)

from argparse import ArgumentParser
import json
import itertools
import os
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from minecraft_attacks_eval_suppress_rule_universal import load_next_state_model_from_wandb, minecraftexp_args_to_wandb_run_name, get_param_dict_list_from_config_file
# import seaborn as sns
import pandas as pd

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

import os
import matplotlib.pyplot as plt

def plot_attention_diffs(attns_without_suffix, attns_with_suffix, figsize=(3, 3), savepdf=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    max_val = 0

    for i in range(len(attns_with_suffix)):
        ax.scatter(attns_with_suffix[i], attns_without_suffix[i], label=f"Sample {sample_i}")
        max_val = max(max_val, max(attns_without_suffix[i]))
        max_val = max(max_val, max(attns_with_suffix[i]))
        # Also plot the y = x line
    
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--")
    ax.set_xlabel("Attns with suffix")
    ax.set_ylabel("Attns without suffix")

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight", transparent=True)
        plt.close()
    return fig

def plot_attention_flow(flow_matrix, token_labels, topk_prefix_start=0, topk_prefix_end=15, savepdf=None, 
                        cbar_text=None,
                        title=None,
                        figsize=(3,2)):
    # flow_matrix = flow_matrix[:, topk_prefix_start:topk_prefix_end]
    orig_token_labels = token_labels
    token_labels = token_labels[topk_prefix_start:topk_prefix_end]
    token_labels[-1] = ". . ."
    token_labels.extend(orig_token_labels[flow_matrix.shape[1] - 4:flow_matrix.shape[1]])
    token_labels[0] = ". . . "
    
    
    indices = list(range(topk_prefix_start, topk_prefix_end))
    indices.extend(list(range(flow_matrix.shape[1] - 4, flow_matrix.shape[1])))
    flow_matrix = flow_matrix[:, indices]
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    # Make up my colors
    blues = colormaps.get_cmap("Blues")
    reds = colormaps.get_cmap("Reds")
    fmin, fmax = flow_matrix.min().item(), flow_matrix.max().item()

    print(f"flow minmax: {fmin:.3f}, {fmax:.3f}")

    
    my_colors = np.vstack([
        np.flip(reds(np.linspace(0, abs(fmin), 32)), axis=0),
        blues(np.linspace(0, abs(fmax), 32)),
        # np.flip(reds(np.linspace(0, 1, 32)), axis=0),
        # blues(np.linspace(0, 1, 32)),
    ])

    my_cmap = LinearSegmentedColormap.from_list("my_colors", my_colors)

    h = ax.pcolor(
        # abs(flow_matrix),
        flow_matrix,
        cmap=my_cmap,
        # vmin=0,
    )
    ax.invert_yaxis()
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis='x', rotation=60)
    ax.set_xticks([0.5 + i for i in range(flow_matrix.shape[1])])
    ax.set_yticks([0.5 + i for i in range(0, flow_matrix.shape[0], 1)])
    ax.set_yticklabels(list(range(1, flow_matrix.shape[0]+1, 1)), fontsize=16)
    ax.set_xticklabels(token_labels, fontsize=16)
    cb = plt.colorbar(h)
    ax.set_ylabel(f"Layers", fontsize=16)
    if title:
        ax.set_title(title)

    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.1f}")
    # Loop over data dimensions and create text annotations.
    for i in range(flow_matrix.shape[0]):
        for j in range(len(token_labels)):
            if abs(flow_matrix[i, j]) < 0.4:
                continue
        # for i in range(1):
            text_color = "w" if abs(flow_matrix[i,j]) > 0.75 else "k" # w=white, k=black

            text = ax.text(j+0.5, i+0.5, valfmt(flow_matrix[i, j]),
                        ha="center", va="center", fontsize=10, color=text_color)
    # else:
    #     ax.set_title("Attention contribution to generation")
    if cbar_text:
        cb.ax.set_title(cbar_text, y=-0.16, fontsize=8)

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    return fig

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
        "--num_tokens_to_generate",
        type=int,
        default=50,
        help="The number of tokens to generate for the experiment.",
    )
    parser.add_argument(
        "--amnesia_rule_depth",
        type=int,
        default=-1,
        help="Depth of the rules to remove for amnesia (if multiple rules as depth, one with randomly be chosen). Default (-1) is the general suppress rule attack"
    )
    parser.add_argument(
        "--suppression_rule_depth",
        type=int,
        default=-1,
        help="Depth of the rules to start suppressing from. Default (-1) is the general suppress rule attack"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to visualize."
    )

    args = parser.parse_args()
    config_file = args.config_file
    num_tokens_to_generate = args.num_tokens_to_generate

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
            artifact_id = f"model-{exp_id}:v0"

            model = AutoModelForCausalLM.from_pretrained("gpt2")
            model = load_next_state_model_from_wandb(artifact_id, model)

            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

            model.to(args.device)
            model.eval()

            if args.amnesia_rule_depth >= 0:
                exp_id = exp_id + f"_amnesia_depth{args.amnesia_rule_depth}"
            elif args.suppression_rule_depth > 0:
                exp_id = exp_id + f"_suppression_depth{args.suppression_rule_depth}"

            # samples_file = os.path.join(args.output_dir, f"{exp_id}_True.json")
            samples_file = os.path.join(args.output_dir, f"{exp_id}.json")
            print(f"Loading samples from {samples_file}")
            # Load samples from the file
            with open(samples_file, "r") as f:
                samples = json.load(f)

                
            all_attns_without_suffix = {
                k : []
                for k in range(12)
            }
            all_attns_with_suffix = {
                k : []
                for k in range(12)
            }
            all_attns_diff = {
                k : []
                for k in range(12)
            }

            for sample_i, example in tqdm(enumerate(samples[:args.num_samples])):
                found_sample = False
                for j, sample in enumerate(example["train_sample_results"][-1::-1]):
                    if sample["attack_success"] and sample["qed"]:
                        found_sample = True
                        break
                if not found_sample:
                    continue
                input_to_model = sample["original_prompt"]
                attack_input_to_model = sample["attack_prompt"]

                tokenized_input = tokenizer(input_to_model, return_tensors="pt").to(args.device)
                tokenized_attack_input = tokenizer(attack_input_to_model, return_tensors="pt").to(args.device)

                facts_start = input_to_model.split("\nI have")[0]
                print(f"Facts start: {facts_start}")
                tokenized_facts_start = len(tokenizer(facts_start, return_tensors="pt").input_ids[0]) + 1 # +1 to ignore the \n

                rules_start = input_to_model.split("If I have")[0]
                tokenized_rules_start = len(tokenizer(rules_start, return_tensors="pt").input_ids[0])

                rules_end = input_to_model.split("Here are some items")[0]
                tokenized_rules_end = len(tokenizer(rules_end, return_tensors="pt").input_ids[0])


                recipe_suppressed = sample["recipe_suppressed"]
                antecedents_suppressed = []
                consequents_suppressed = []

                suppressed_rule_token_pos = []

                for rule in recipe_suppressed:
                    if rule[2] == param_instance["num_steps"] - 1:
                        antecedents = rule[0]
                        consequent_suppressed = rule[1]
                        antecedents = [a.replace("minecraft:", "").replace("_", " ") for a in antecedents]
                        consequent_suppressed = consequent_suppressed.replace("minecraft:", "").replace("_", " ")
                        antecedents_suppressed.extend(antecedents)

                        print(recipe_suppressed)
                        print(antecedents)
                        print(consequent_suppressed)

                        suppresed_rule_pos = [i for i, rule in enumerate(input_to_model.split(".")) if f"then I can create {consequent_suppressed}" in rule and all([ante in rule for ante in antecedents])]

                        print(suppresed_rule_pos)

                        assert len(suppresed_rule_pos) == 1

                        suppressed_rule_start = ".".join(input_to_model.split(".")[:suppresed_rule_pos[0]]) + "."
                        print(f"Suppressed rule start: {suppressed_rule_start}")

                        tokenized_suppressed_rule_start = len(tokenizer(suppressed_rule_start, return_tensors="pt").input_ids[0])

                        suppressed_rule_end = ".".join(input_to_model.split(".")[:suppresed_rule_pos[0]+1])
                        print(f"Suppressed rule end: {suppressed_rule_end}")

                        tokenized_suppressed_rule_end = len(tokenizer(suppressed_rule_end, return_tensors="pt").input_ids[0])

                        suppressed_rule_token_pos.extend(list(range(tokenized_suppressed_rule_start, tokenized_suppressed_rule_end)))

                sample_attns_without_suffix = {
                    k : 0
                    for k in range(12)
                }
                sample_attns_with_suffix = {
                    k : 0
                    for k in range(12)
                }
                sample_attns_diff = {
                    k : -999
                    for k in range(12)
                }

                tks_without = []
                tks_with = []

                # Analyze attentions of all layers with and without the adversarial suffix
                with torch.no_grad():
                    model.config.output_attentions = True

                    op_without_suffix = model.generate(**tokenized_input, max_new_tokens =num_tokens_to_generate, output_attentions=True, return_dict_in_generate=True, do_sample=False)
                    op_with_suffix = model.generate(**tokenized_attack_input, max_new_tokens=num_tokens_to_generate, output_attentions=True, return_dict_in_generate=True, do_sample=False)

                    for ntk in tqdm(range(num_tokens_to_generate)):
                        print(f"Sample_i: {sample_i}")
                        tokens_without_suffix = [tokenizer.decode(t) for t in op_without_suffix.sequences[0][:-(num_tokens_to_generate-ntk)]]
                        tokens_with_suffix = [tokenizer.decode(t) for t in op_with_suffix.sequences[0][:-(num_tokens_to_generate-ntk)]]
                        print(f"Shape of tokens with suffix: {len(tokens_with_suffix)}")
                        print(f"Shape of tokens without suffix: {len(tokens_without_suffix)}")

                        tks_without.append(tokenizer.decode(op_without_suffix.sequences[0][len(tokenized_input.input_ids[0])+ntk]))
                        tks_with.append(tokenizer.decode(op_with_suffix.sequences[0][len(tokenized_attack_input.input_ids[0])+ntk]))

                        ante_pos = [(i, tokens_with_suffix[i]) for i in suppressed_rule_token_pos]
                        print(f"Final ante_pos: {ante_pos}")

                        ante_pos = [a[0] for a in ante_pos]
                        # ante_pos = [0]

                        attns_without_suffix = op_without_suffix.attentions[ntk]
                        attns_with_suffix = op_with_suffix.attentions[ntk]

                        # Get the attentions for the last token
                        attns_without_suffix = [attns_without_suffix[layer].squeeze(0) for layer in range(len(attns_without_suffix))]
                        attns_with_suffix = [attns_with_suffix[layer].squeeze(0) for layer in range(len(attns_with_suffix))]

                        print(attns_without_suffix[0].shape)

                        attns_without_suffix = [attns_without_suffix_layer.max(dim=0).values for attns_without_suffix_layer in attns_without_suffix]
                        attns_with_suffix = [attns_with_suffix_layer.max(dim=0).values for attns_with_suffix_layer in attns_with_suffix]

                        # Get the attentions for the last token
                        attns_without_suffix = [attns_without_suffix_layer[-1] for attns_without_suffix_layer in attns_without_suffix]
                        attns_with_suffix = [attns_with_suffix_layer[-1] for attns_with_suffix_layer in attns_with_suffix]
                        
                        # Generate the heatmap

                        heatmap_without_suffix = torch.stack(attns_without_suffix).cpu().numpy()
                        heatmap_with_suffix = torch.stack(attns_with_suffix).cpu().numpy()

                        # Switch the axes
                        heatmap_without_suffix = heatmap_without_suffix.transpose(1, 0)
                        heatmap_with_suffix = heatmap_with_suffix.transpose(1, 0) 

                        # Pad the heatmap without suffix
                        heatmap_without_suffix_padded = np.zeros(heatmap_with_suffix.shape)
                        heatmap_without_suffix_padded[:heatmap_without_suffix.shape[0], :heatmap_without_suffix.shape[1]] = heatmap_without_suffix

                        # Subtract the two heatmaps
                        heatmap_diff = heatmap_with_suffix - heatmap_without_suffix_padded
                        # heatmap_diff = heatmap_without_suffix_padded - heatmap_with_suffix

                        heatmap_diff = heatmap_diff.transpose(1, 0)
                        # Switch the axes
                        heatmap_without_suffix = heatmap_without_suffix.transpose(1, 0)
                        heatmap_with_suffix = heatmap_with_suffix.transpose(1, 0) 


                        # heatmap_diff = heatmap_without_suffix_padded - heatmap_with_suffix

                        print(f"Shape of heatmap_diff: {heatmap_diff.shape}")

                        # Heatmap diff for antes
                        heatmap_diff_ante = heatmap_diff[:, ante_pos]

                        print(f"Shape of heatmap_diff_ante: {heatmap_diff_ante.shape}")
                        print(f"Token with: {tks_with[ntk]}")
                        print(f"Token without: {tks_without[ntk]}")
                        print(f"Heatmap diff ante: {np.max(heatmap_diff_ante)}")
                        print(f"Attn without: {np.max(heatmap_without_suffix[:, ante_pos])}")
                        print(f"Attn with: {np.max(heatmap_with_suffix[:, ante_pos])}")

                        for l in range(12):
                            sample_attns_with_suffix[l] = max(sample_attns_with_suffix[l], np.max(heatmap_with_suffix[l, ante_pos]))
                            sample_attns_without_suffix[l] = max(sample_attns_without_suffix[l], np.max(heatmap_without_suffix[l, ante_pos]))
                            sample_attns_diff[l] = max(sample_attns_diff[l], np.max(heatmap_diff[l, ante_pos]))

                        fig = plot_attention_flow(
                            heatmap_diff, 
                            tokens_with_suffix, 
                            savepdf=f"attn_test_suppression1/attention_test/{sample_i}/{ntk}_diff",
                            figsize=(18, 4),
                            # figsize=(100, 6),
                            # figsize=(48, 6),
                            # topk_prefix_start=0,
                            # topk_prefix_start=tokenized_facts_start-1,
                            topk_prefix_start=tokenized_rules_start,
                            topk_prefix_end=tokenized_rules_end
                            # topk_prefix_end=heatmap_diff.shape[1]
                            )
                        
                        fig = plot_attention_flow(
                            abs(heatmap_diff), 
                            tokens_with_suffix, 
                            savepdf=f"attn_test_suppression1/attention_test/{sample_i}/{ntk}_diff_abs",
                            figsize=(18, 4),
                            # figsize=(3, 24),
                            # figsize=(100, 6),
                            # figsize=(48, 6),
                            # topk_prefix_start=-86,
                            # topk_prefix_start=tokenized_facts_start-1,
                            topk_prefix_start=tokenized_rules_start,
                            topk_prefix_end=tokenized_rules_end
                            # topk_prefix_end=heatmap_diff.shape[1]
                            )
                        
                        fig = plot_attention_flow(
                            heatmap_with_suffix, 
                            tokens_with_suffix, 
                            savepdf=f"attn_test_suppression1/attention_test/{sample_i}/{ntk}_with_suffix",
                            figsize=(18, 4),
                            # figsize=(3, 24),
                            # figsize=(100, 6),
                            # figsize=(48, 6),
                            # topk_prefix_start=-86,
                            # topk_prefix_start=tokenized_facts_start-1,
                            topk_prefix_start=tokenized_rules_start,
                            topk_prefix_end=tokenized_rules_end
                            # topk_prefix_end=heatmap_diff.shape[1]
                            )
                        
                        fig = plot_attention_flow(
                            heatmap_without_suffix, 
                            tokens_without_suffix, 
                            savepdf=f"attn_test_suppression1/attention_test/{sample_i}/{ntk}_without_suffix",
                            figsize=(18, 4),
                            # figsize=(3, 24),
                            # figsize=(100, 6),
                            # figsize=(48, 6),
                            # topk_prefix_start=-86,
                            # topk_prefix_start=tokenized_facts_start-1,
                            topk_prefix_start=tokenized_rules_start,
                            topk_prefix_end=tokenized_rules_end
                            # topk_prefix_end=heatmap_diff.shape[1]
                            )
                        
                print(f"Tokens with suffix: {tokens_with_suffix}")
                print(f"Tokens without suffix: {tokens_without_suffix}")

                d = {"Reasoner": [f"t={param_instance['num_steps']}", ""], "Att. Wt.": ["Without suffix", "With suffix"]}
                for l in range(12):
                    all_attns_with_suffix[l].append(sample_attns_with_suffix[l])
                    all_attns_without_suffix[l].append(sample_attns_without_suffix[l])
                    all_attns_diff[l].append(sample_attns_diff[l])

                    print(f"Layer {l}")

                    print(f"Num samples: {len(all_attns_with_suffix[l])}")
                    print(f"All attns with suffix (mean: {sum(all_attns_with_suffix[l])/len(all_attns_with_suffix[l])}): {all_attns_with_suffix[l]}")
                    print(f"All attns without suffix (mean: {sum(all_attns_without_suffix[l])/len(all_attns_without_suffix[l])}): {all_attns_without_suffix[l]}")
                    print(f"All attns diff (mean: {sum(all_attns_diff[l])/len(all_attns_diff[l])}): {all_attns_diff[l]}")

                    print("\n---------------------\n")

                    d[f"{l}"] = [sum(all_attns_without_suffix[l])/len(all_attns_without_suffix[l]), sum(all_attns_with_suffix[l])/len(all_attns_with_suffix[l])]
                df = pd.DataFrame(d)

                print(df.to_latex(index=False, float_format="{:.2f}".format,))
