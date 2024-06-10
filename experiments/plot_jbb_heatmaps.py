import sys
from pathlib import Path
from common import *
from argparse import ArgumentParser
import os
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json

import matplotlib.pyplot as plt

import jailbreakbench as jbb

from transformers import utils
utils.logging.set_verbosity_error()  # Suppress standard warnings

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

VICUNA_HF_CHAT_TEMPLATE = """\
{% for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{ message['content'].strip() + ' ' }}
    {%- elif message['role'] == 'user' -%}
        {{ 'USER: ' + message['content'].strip() + ' ' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ 'ASSISTANT: '  + message['content'] + eos_token + '' }}
    {%- endif %}
    {%- if loop.last and add_generation_prompt -%}
        {{ 'ASSISTANT:' }}
    {%- endif -%}
{%- endfor %}
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama-2-7b-chat-hf",
        choices=["llama-2-7b-chat-hf", "vicuna-13b-v1.5"],
        help="The model to use for the experiment.",
    )
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=5,
        help="The maximum number of samples to consider.",
    )
    parser.add_argument(
        "--num_tokens_to_generate",
        type=int,
        default=50,
        help="The number of tokens to generate for the experiment.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="The device to use for the experiment.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attention_heatmaps_jbb_gcg_removelastsys",
        help="The output directory to save the attention heatmaps.",
    )
    args = parser.parse_args()
    model_name = args.model_name
    max_num_samples = args.max_num_samples
    output_dir = args.output_dir
    num_tokens_to_generate = args.num_tokens_to_generate
    device = torch.device(args.device)
    num_fig_cols = 4
    # model_id = HF_MODEL_NAMES[MODEL_NAME]
    if "llama" in model_name:
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        # system_prompt = "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        # system_prompt = ""
        # system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
        num_layers = 32
        num_heads = 32
        num_fig_rows = 8
    elif "vicuna" in model_name:
        model_id = f"lmsys/vicuna-13b-v1.5"
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        system_prompt = "You are a helpful, respectful and honest assistant. Please ensure that you do not compromise the privacy of others. This includes refraining from sharing personal information, leaking confidential data, or engaging in any behavior that may violate someone else's privacy rights."
        system_prompt = "You are a helpful, respectful and honest assistant. Please ensure that you do not compromise the privacy of others. This includes refraining from sharing personal information, leaking confidential data, or engaging in any behavior that may violate someone else's privacy rights. Do not provide any assistance in indirectly compromising someone else's privacy."
        # system_prompt = "Please ensure that you do not compromise the privacy of others. This includes refraining from sharing personal information, leaking confidential data, or engaging in any behavior that may violate someone else's privacy rights. Do not provide any assistance in indirectly compromising someone else's privacy."
        num_layers = 40
        num_heads = 40
        num_fig_rows = 10
    
    model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    print(f"Model dtype: {model.dtype}")

    if "vicuna" in model_name:
        tokenizer.chat_template = VICUNA_HF_CHAT_TEMPLATE

    model.to(device)
    model.eval()

    attack_stats = {
        k : [] for k in range(num_layers)
    }

    attack_stats_head = {
        k: {
            l: [] for l in range(num_heads)
        } for k in range(num_layers)
    }

    system_stats = {
        k : [] for k in range(num_layers)
    }

    system_stats_attack = {
        k : [] for k in range(num_layers)
    }

    artifact = jbb.read_artifact(method="GCG", model_name=model_name)
    # samples = [j for j in artifact.jailbreaks if j.jailbroken]
    samples = [j for j in artifact.jailbreaks if j.jailbroken and "Privacy" in j.category]

    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    with open(f"{output_dir}/{model_name}/info.json", "w") as f:
        info = {
            "model_name": model_name,
            "model_id": model_id,
            "model_dtype": str(model.dtype),
            "system_prompt": system_prompt,
            "max_num_samples": max_num_samples,
            "num_tokens_to_generate": num_tokens_to_generate,
            "tokenizer_chat_template": tokenizer.chat_template,
            "jbb_artifact_method": "GCG",
            "samples": [dict(sample) for sample in samples]
        }
        json.dump(info, f)

    attack_samples = [sample.prompt for sample in samples[:max_num_samples]]
    samples = [sample.goal for sample in samples[:max_num_samples]]
    
    for sample_i, sample in tqdm(enumerate(samples)):
        input_to_model = sample
        attack_input_to_model = attack_samples[sample_i]

        # This is to track how much attention is given to the Llama system prompt
        # Change this later
        system_input_to_model = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>"

        input_to_model = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": input_to_model}]
        
        attack_input_to_model = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": attack_input_to_model}
        ]

        print(f"Generation configuration: {model.generation_config}")

        print(f"Input to model: {input_to_model}\n--------\n")
        tokenized_input = tokenizer.apply_chat_template(input_to_model, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        tokenized_attack_input = tokenizer.apply_chat_template(attack_input_to_model, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        tokenized_system_input = tokenizer(system_input_to_model, return_tensors="pt").input_ids.to(device)

        print(f"Tokenized attack input.shape: {tokenized_attack_input.shape}")
        print(f"Tokenized system input.shape: {tokenized_system_input.shape}")     
        print(f"Decoded system input: {tokenizer.decode(tokenized_system_input[0][9:])}") 

        original_tokenized_input = tokenized_input.clone()
        original_tokenized_attack_input = tokenized_attack_input.clone()

        # Analyze attentions of all layers with and without the adversarial suffix
        with torch.no_grad():
            model.config.output_attentions = True

            op_without_suffix = model.generate(tokenized_input, max_new_tokens =num_tokens_to_generate, output_attentions=True, return_dict_in_generate=True, do_sample=False)
            op_with_suffix = model.generate(tokenized_attack_input, max_new_tokens=num_tokens_to_generate, output_attentions=True, return_dict_in_generate=True, do_sample=False)

            print(f"Decoded without suffix: {tokenizer.decode(op_without_suffix.sequences[0])}")
            print(f"Decoded with suffix: {tokenizer.decode(op_with_suffix.sequences[0])}")

            print("\n++++++++++++++++++++++++++++++++\n")

            # Get list of tokens for input and attack input
            tokens_without_suffix = tokenizer.convert_ids_to_tokens(op_without_suffix.sequences[0])
            tokens_with_suffix = tokenizer.convert_ids_to_tokens(op_with_suffix.sequences[0])

            for ntk in tqdm(range(num_tokens_to_generate)):
                attns_without_suffix = op_without_suffix.attentions[ntk]
                attns_with_suffix = op_with_suffix.attentions[ntk]

                orig_attns_without_suffix = attns_without_suffix
                orig_attns_with_suffix = attns_with_suffix

                # print(f"Shape of attns_without_suffix: {attns_without_suffix[0]}")

                # Create a heatmap of the attention scores for the last token
                # Sum the attention scores for all heads in a layer
                # The heatmap's x-axis is the layer number
                # The heatmap's y-xis is the token value

                # Get the attentions for the last token
                attns_without_suffix = [attns_without_suffix[layer].squeeze(0) for layer in range(len(attns_without_suffix))]
                attns_with_suffix = [attns_with_suffix[layer].squeeze(0) for layer in range(len(attns_with_suffix))]

                attns_without_suffix = [attns_without_suffix_layer.sum(dim=0) for attns_without_suffix_layer in attns_without_suffix]
                attns_with_suffix = [attns_with_suffix_layer.sum(dim=0) for attns_with_suffix_layer in attns_with_suffix]

                # Get the attentions for the last token
                attns_without_suffix = [attns_without_suffix_layer[-1] for attns_without_suffix_layer in attns_without_suffix]
                attns_with_suffix = [attns_with_suffix_layer[-1] for attns_with_suffix_layer in attns_with_suffix]

                # Generate the heatmap

                heatmap_without_suffix = torch.stack(attns_without_suffix).cpu().numpy()
                heatmap_with_suffix = torch.stack(attns_with_suffix).cpu().numpy()

                # Switch the axes
                heatmap_without_suffix = heatmap_without_suffix.transpose(1, 0)
                heatmap_with_suffix = heatmap_with_suffix.transpose(1, 0) 

                if "llama" in model_name:
                    fig, axs = plt.subplots(1, 2, figsize=(25, 60))
                elif "vicuna" in model_name:
                    fig, axs = plt.subplots(1, 2, figsize=(30, 60))

                axs[0].set_title("Attention Heatmap without Suffix")
                axs[1].set_title("Attention Heatmap with Suffix")

                # Plot the heatmaps
                axs[0].imshow(heatmap_without_suffix, cmap="RdYlGn", interpolation="nearest")
                axs[1].imshow(heatmap_with_suffix, cmap="RdYlGn", interpolation="nearest")

                # Label the axes
                axs[0].set_xlabel("Layer")
                axs[0].set_ylabel("Token")

                # Use tokens as the y-axis
                axs[0].set_yticks(range(len(tokens_without_suffix)))
                axs[0].set_yticklabels(tokens_without_suffix)
                axs[0].set_xticks(range(num_layers))
                axs[0].set_xticklabels(range(num_layers))

                axs[1].set_xlabel("Layer")
                axs[1].set_ylabel("Token")

                # Use tokens as the y-axis
                axs[1].set_yticks(range(len(tokens_with_suffix)))
                axs[1].set_yticklabels(tokens_with_suffix)
                axs[1].set_xticks(range(num_layers))
                axs[1].set_xticklabels(range(num_layers))

                # Add a colorbar
                fig.colorbar(axs[0].imshow(heatmap_without_suffix, cmap="RdYlGn", interpolation="nearest"), ax=axs[0])
                fig.colorbar(axs[1].imshow(heatmap_with_suffix, cmap="RdYlGn", interpolation="nearest"), ax=axs[1])


                os.makedirs(f"{output_dir}/{model_name}/sample_{sample_i}", exist_ok=True)
                print("Saving figure as ", f"{output_dir}/{model_name}/sample_{sample_i}/token_{ntk}.png")
                plt.savefig(f"{output_dir}/{model_name}/sample_{sample_i}/token_{ntk}.png")

                print(heatmap_with_suffix.shape)
                print(heatmap_without_suffix.shape)
                
                if "llama" in model_name:
                    inst_tokens = 4 + ntk   # 4 ending INST tokens
                elif "vicuna" in model_name:
                    inst_tokens = 5 + ntk   # 5 ending ASSISTANT tokens

                # Plot the difference
                # Pad the heatmap without suffix
                heatmap_without_suffix_padded = np.zeros(heatmap_with_suffix.shape)
                heatmap_without_suffix_padded[:(heatmap_without_suffix.shape[0]-inst_tokens), :] = heatmap_without_suffix[:(heatmap_without_suffix.shape[0]-inst_tokens), :]
                heatmap_without_suffix_padded[(heatmap_with_suffix.shape[0]-inst_tokens):, :] = heatmap_without_suffix[(heatmap_without_suffix.shape[0]-inst_tokens):, :]

                # Subtract the two heatmaps
                heatmap_diff = heatmap_with_suffix - heatmap_without_suffix_padded
                print(f"Padded:")
                print(heatmap_without_suffix_padded[(heatmap_without_suffix.shape[0]-inst_tokens):(heatmap_with_suffix.shape[0]-inst_tokens), :])

                fig, axs = plt.subplots(1, 1, figsize=(25, 60))
                axs.set_title("Attention Heatmap Difference")
                axs.imshow(heatmap_diff, cmap="RdYlGn", interpolation="nearest")

                # Label the axes
                axs.set_xlabel("Layer")
                axs.set_ylabel("Token")

                # Use tokens as the y-axis
                axs.set_yticks(range(len(tokens_with_suffix)))
                axs.set_yticklabels(tokens_with_suffix)

                axs.set_xticks(range(num_layers))
                axs.set_xticklabels(range(num_layers))

                # Add a colorbar
                fig.colorbar(axs.imshow(heatmap_diff, cmap="RdYlGn", interpolation="nearest"), ax=axs)

                plt.savefig(f"{output_dir}/{model_name}/sample_{sample_i}/token_{ntk}_diff.png")

                for layer in range(len(attns_without_suffix)):
                    attns_without_suffix_layer = orig_attns_without_suffix[layer].squeeze(0)
                    attns_with_suffix_layer = orig_attns_with_suffix[layer].squeeze(0)

                    attns_without_suffix_layer_avg = attns_without_suffix_layer.mean(dim=0)
                    attns_with_suffix_layer_avg = attns_with_suffix_layer.mean(dim=0)

                    attack_volume = (tokenized_attack_input.shape[-1] - original_tokenized_input.shape[-1]) / tokenized_attack_input.shape[-1]
                    attack_leverage = attns_with_suffix_layer_avg[-1][original_tokenized_input.shape[-1]:].sum() / attns_with_suffix_layer_avg[-1].sum()
                    attack_leverage = attack_leverage.item()
                    attack_stats[layer].append((sample_i, ntk, attack_volume, attack_leverage))

                    system_volume_attack = (tokenized_system_input.shape[-1] - 9) / tokenized_attack_input.shape[-1]
                    system_leverage_attack = attns_with_suffix_layer_avg[-1][9:tokenized_system_input.shape[-1]].sum() / attns_with_suffix_layer_avg[-1].sum()
                    system_leverage_attack = system_leverage_attack.item()

                    system_volume = (tokenized_system_input.shape[-1] - 9) / original_tokenized_input.shape[-1]
                    system_leverage = attns_without_suffix_layer_avg[-1][9:tokenized_system_input.shape[-1]].sum() / attns_without_suffix_layer_avg[-1].sum()
                    system_leverage = system_leverage.item()

                    system_stats[layer].append((sample_i, ntk, system_volume, system_leverage))
                    system_stats_attack[layer].append((sample_i, ntk, system_volume_attack, system_leverage_attack))

                    for head in range(attns_without_suffix_layer.shape[0]):
                        attns_without_suffix_head = attns_without_suffix_layer[head]
                        attns_with_suffix_head = attns_with_suffix_layer[head]

                        attack_leverage = attns_with_suffix_head[-1][original_tokenized_input.shape[-1]:].sum() / attns_with_suffix_head[-1].sum()
                        attack_leverage = attack_leverage.item()

                        attack_stats_head[layer][head].append((sample_i, ntk, attack_volume, attack_leverage))


    # exit()
                        
    # Plot the system stats such that each number of tokens is plotted as a separate color
    # Each plot has (attack volume, attack leverage) as points on the graph
    fig, axs = plt.subplots(num_fig_cols, num_fig_rows, figsize=(20, 20))

    for layer in tqdm(range(num_layers), desc="Plotting system stats for each layer"):
        if len(system_stats[layer]) == 0:
            continue

        axs[layer // num_fig_rows, layer % num_fig_rows].set_title(f"Layer: {layer}")

        system_stats_layer = system_stats[layer]
        system_stats_layer = np.array(system_stats_layer)

        # Each sample_i is a different color
        for sample_i in range(len(samples)):
            sample_i_indices = np.where(system_stats_layer[:, 0] == sample_i)
            sample_i_data = system_stats_layer[sample_i_indices]
            axs[layer // num_fig_rows, layer % num_fig_rows].scatter(sample_i_data[:, 2], sample_i_data[:, 3], label=f"Sample {sample_i} (without suffix)")

        system_stats_attack_layer = system_stats_attack[layer]
        system_stats_attack_layer = np.array(system_stats_attack_layer)

        # Each sample_i is a different color
        for sample_i in range(len(samples)):
            sample_i_indices = np.where(system_stats_attack_layer[:, 0] == sample_i)
            sample_i_data = system_stats_attack_layer[sample_i_indices]
            axs[layer // num_fig_rows, layer % num_fig_rows].scatter(sample_i_data[:, 2], sample_i_data[:, 3], label=f"Sample {sample_i} (with suffix)")

        # Also plot the y = x line
        axs[layer // num_fig_rows, layer % num_fig_rows].plot([0, 1], [0, 1], color="red", linestyle="--")

    # Add a legend
    axs[0, 0].legend()

    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    plt.savefig(f"{output_dir}/{model_name}/system_stats_{model_name}.png")

    # Plot the attack stats such that each number of tokens is plotted as a separate color
    # Each plot has (attack volume, attack leverage) as points on the graph
    fig, axs = plt.subplots(num_fig_cols, num_fig_rows, figsize=(20, 20))

    for layer in tqdm(range(num_layers), desc="Plotting attack stats for each layer"):
        if len(attack_stats[layer]) == 0:
            continue

        attack_stats_layer = attack_stats[layer]
        attack_stats_layer = np.array(attack_stats_layer)

        axs[layer // num_fig_rows, layer % num_fig_rows].set_title(f"Layer: {layer}")
        # Each sample_i is a different color
        for sample_i in range(len(samples)):
            sample_i_indices = np.where(attack_stats_layer[:, 0] == sample_i)
            sample_i_data = attack_stats_layer[sample_i_indices]
            axs[layer // num_fig_rows, layer % num_fig_rows].scatter(sample_i_data[:, 2], sample_i_data[:, 3], label=f"Sample {sample_i}")

        # Also plot the y = x line
        axs[layer // num_fig_rows, layer % num_fig_rows].plot([0, 1], [0, 1], color="red", linestyle="--")

    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    plt.savefig(f"{output_dir}/{model_name}/attack_stats_{model_name}.png")

    for layer in tqdm(range(num_layers), desc="Plotting attack stats for each layer and head"):
        if len(attack_stats_head[layer][0]) == 0:
            continue

        fig, axs = plt.subplots(num_fig_cols, num_fig_rows, figsize=(20, 20))
        for head in range(num_heads):
            if len(attack_stats_head[layer][head]) == 0:
                continue

            attack_stats_head_layer_head = attack_stats_head[layer][head]
            attack_stats_head_layer_head = np.array(attack_stats_head_layer_head)

            axs[head // num_fig_rows, head % num_fig_rows].set_title(f"Layer: {layer}, Head: {head}")
            # Each sample_i is a different color
            for sample_i in range(len(samples)):
                sample_i_indices = np.where(attack_stats_head_layer_head[:, 0] == sample_i)
                sample_i_data = attack_stats_head_layer_head[sample_i_indices]
                axs[head // num_fig_rows, head % num_fig_rows].scatter(sample_i_data[:, 2], sample_i_data[:, 3], label=f"Sample {sample_i}")

            # Also plot the y = x line
            axs[head // num_fig_rows, head % num_fig_rows].plot([0, 1], [0, 1], color="red", linestyle="--")

        os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
        plt.savefig(f"{output_dir}/{model_name}/attack_stats_{model_name}_layer_{layer}.png")