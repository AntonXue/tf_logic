import sys
from pathlib import Path
from argparse import ArgumentParser
import os
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

import pandas as pd

from hooks import TraceDict

# python experiments/plot_naive_heatmaps.py --output_dir llama-new_att

from transformers import utils
utils.logging.set_verbosity_error()  # Suppress standard warnings

from dotenv import load_dotenv

load_dotenv()

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

def plot_attention_flow(flow_matrix, token_labels, topk_prefix_start=0, topk_prefix_end=15, savepdf=None, 
                        cbar_text=None,
                        title=None,
                        figsize=(3,2)):
    flow_matrix = flow_matrix[:, topk_prefix_start:topk_prefix_end]
    token_labels = token_labels[topk_prefix_start:topk_prefix_end]
    token_labels[0] = ". . . "
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    # Make up colors
    blues = colormaps.get_cmap("Blues")
    reds = colormaps.get_cmap("Reds")
    fmin, fmax = flow_matrix.min().item(), flow_matrix.max().item()

    if fmin < 0:
        my_colors = np.vstack([
            np.flip(reds(np.linspace(0, abs(fmin), 32)), axis=0),
            blues(np.linspace(0, abs(fmax), 32))
        ])
    else:
        my_colors = blues(np.linspace(0, abs(fmax), 32))

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
    cb.ax.tick_params(labelsize=12)

    ax.set_ylabel(f"Layers", fontsize=16)
    if title:
        ax.set_title(title)

    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.1f}")
    # Loop over data dimensions and create text annotations.
    for i in range(flow_matrix.shape[0]):
        for j in range(len(token_labels)):
            if abs(flow_matrix[i, j]) < 0.5:
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

    save_dict = {
        "flow_matrix": flow_matrix,
        "tokens": token_labels
    }

    torch.save(save_dict, savepdf + ".pt")

    return fig

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
        "--dataset_json_path",
        type=str,
        default="./rule_following.json",
        help="The json file containing the dataset.",
    )
    parser.add_argument(
        "--num_tokens_to_generate",
        type=int,
        default=15,
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
        default="attention_heatmaps_jbb",
        help="The output directory to save the attention heatmaps.",
    )
    args = parser.parse_args()
    model_name = args.model_name
    output_dir = args.output_dir
    num_tokens_to_generate = args.num_tokens_to_generate
    device = torch.device(args.device)
    num_fig_cols = 4
    # model_id = HF_MODEL_NAMES[MODEL_NAME]
    if "llama" in model_name:
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        system_prompt = "If a question is about sweet potatoes, say 'I only know about regular potatoes' and terminate the conversation. If a question is about oranges, answer with what you know about that topic."
        num_layers = 32
        num_heads = 32
        num_fig_rows = 8
    elif "vicuna" in model_name:
        model_id = f"lmsys/vicuna-7b-v1.5"
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        num_layers = 40
        num_heads = 40
        num_fig_rows = 10
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(model.config.torch_dtype)

    if "vicuna" in model_name:
        tokenizer.chat_template = VICUNA_HF_CHAT_TEMPLATE

    model.to(device)
    model.eval()

    attack_stats = {
        k : [] for k in range(num_layers)
    }

    orig_attack_stats = {
        k : [] for k in range(num_layers)
    }

    attack_stats_head = {
        k: {
            l: [] for l in range(num_heads)
        } for k in range(num_layers)
    }

    orig_attack_stats_head = {
        k: {
            l: [] for l in range(num_heads)
        } for k in range(num_layers)
    }

    all_attns_without_suffix = {
        k : []
        for k in range(num_layers)
    }
    all_attns_with_suffix = {
        k : []
        for k in range(num_layers)
    }
    all_attns_diff = {
        k : []
        for k in range(num_layers)
    }

    model_config = model.model.config
    num_key_value_groups = model_config.num_attention_heads // model_config.num_key_value_heads
    attention_layers = [(i, f"layers.{i}.self_attn") for i in range(model.model.config.num_hidden_layers)]
    # mlp_layers = [(i, f"model.layers.{i}.mlp") for i in range(model_wrapped.model.config.num_hidden_layers)]
    all_layers = attention_layers

    samples = json.load(open(args.dataset_json_path))

    print(f"Found {len(samples)} samples.")

    if "llama" in model_name:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    
    for sample_i, sample in tqdm(enumerate(samples)):
        input_to_model = [
            # {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": f"{sample['system_prompt']} {sample['user_prompt']}"}]
        
        attack_input_to_model = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{sample['system_prompt']} {sample['user_prompt']} {sample['adversarial_suffix']}"}
        ]

        sample_attns_without_suffix = {
            k : 0
            for k in range(num_layers)
        }
        sample_attns_with_suffix = {
            k : 0
            for k in range(num_layers)
        }
        sample_attns_diff = {
            k : -999
            for k in range(num_layers)
        }

        # input_to_model = f"{system_prompt}{input_to_model}"
        # attack_input_to_model = f"{system_prompt}{attack_input_to_model}"

        

        # input_to_model = [
        #     {"role": "system", "content": system_prompt}, 
        #     {"role": "user", "content": f"{input_to_model}"}]
        
        # attack_input_to_model = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": f"{attack_input_to_model}"}
        # ]

        # print(f"Generation configuration: {model.generation_config}")

        print(f"Input to model: {input_to_model}\n--------\n")
        tokenized_input = tokenizer.apply_chat_template(input_to_model, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        tokenized_attack_input = tokenizer.apply_chat_template(attack_input_to_model, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

        # tokenized_input = tokenizer(input_to_model, return_tensors="pt").to(device)
        # tokenized_attack_input = tokenizer(attack_input_to_model, return_tensors="pt").to(device)

        # print(f"Tokenized input: {tokenized_input.shape}")
        # print(f"Output: {model.generate(tokenized_input, max_new_tokens=1, do_sample=False, temperature=0, top_p=1, output_attentions=True, output_hidden_states=True, return_dict_in_generate=True).sequences.shape}")
        # exit()

        original_tokenized_input = tokenized_input.clone()
        original_tokenized_attack_input = tokenized_attack_input.clone()

        # Analyze attentions of all layers with and without the adversarial suffix
        with torch.no_grad():
            # model.generation_config.output_attentions = True
            # model.generation_config.return_dict_in_generate = True
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.pad_token_id=tokenizer.pad_token_id

            print(model.generation_config)

            # op_without_suffix = model.generate(tokenized_input, output_attentions=True, return_dict_in_generate=True, do_sample=False, temperature=None, top_p=None, max_new_tokens=25)
            # op_without_suffix = model.generate(tokenized_input, do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.pad_token_id, output_attentions=True, return_dict_in_generate=True)
            # op_with_suffix = model.generate(tokenized_attack_input, do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.pad_token_id, output_attentions=True, return_dict_in_generate=True)

            op_without_suffix = model.generate(tokenized_input, output_hidden_states=True, output_attentions=True)
            op_with_suffix = model.generate(tokenized_attack_input, output_hidden_states=True, output_attentions=True)

            print(model.generation_config)

            print(f"Without suffix: {tokenizer.decode(op_without_suffix[0])}")
            print(f"With suffix: {tokenizer.decode(op_with_suffix[0])}")

            # continue

            tokens_without_suffix = [tokenizer.decode(t) for t in op_without_suffix[0]]
            print(f"tokens without suffix: {tokens_without_suffix}")
            # tokens_with_suffix = [tokenizer.decode(t) for t in op_with_suffix[0]]

            rule = sample['rule_suppressed'].replace(" ", "")

            
            constructed_op = ""
            for a in range(len(tokens_without_suffix)):
                found = False
                for b in range(a+1, len(tokens_without_suffix)):
                    constructed_op = "".join(tokens_without_suffix[a:b])
                    # print(constructed_op)
                    if rule == constructed_op:
                        print(f"Found rule")
                        rule_start = a
                        rule_end = b
                        found = True
                        break
                if found:
                    break
            ante_pos = list(range(rule_start, rule_end))
            print(f"Ante pos: {ante_pos}")
            print(f"Tokens: {tokens_without_suffix[rule_start:rule_end]}")

            

            # continue
            # op_with_suffix = model.generate(tokenized_attack_input, output_attentions=True, return_dict_in_generate=True, do_sample=False, temperature=None, top_p=None, max_new_tokens=25)

            # op_without_suffix = model.generate(**tokenized_input, output_attentions=True, return_dict_in_generate=True, do_sample=False, temperature=None, top_p=None, max_new_tokens=15)
            # op_with_suffix = model.generate(**tokenized_attack_input, output_attentions=True, return_dict_in_generate=True, do_sample=False, temperature=None, top_p=None, max_new_tokens=15)

            # print(f"Decoded without suffix: {tokenizer.decode(op_without_suffix.sequences[0])}")
            # print(f"Decoded with suffix: {tokenizer.decode(op_with_suffix.sequences[0])}")

            # continue

            # Get list of tokens for input and attack input
            # tokens_without_suffix = tokenizer.convert_ids_to_tokens(op_without_suffix.sequences[0])
            # tokens_with_suffix = tokenizer.convert_ids_to_tokens(op_with_suffix.sequences[0])
                        

            for ntk in tqdm(range(num_tokens_to_generate)):
                print(ntk)

                tokenized_input = op_without_suffix[:, :original_tokenized_input.shape[1]+ntk]
                tokenized_attack_input = op_with_suffix[:, :original_tokenized_attack_input.shape[1]+ntk]

                print(f"Shape of original tokenized input: {original_tokenized_input.shape}")
                print(f"Shape of new tokenized_inp: {tokenized_input.shape}")

                with torch.no_grad(), TraceDict(model.model, [l[1] for l in all_layers]) as ret:
                    # print(f"Ret keys: {ret.keys()}")
                    op_without_ntk = model.model(tokenized_input, torch.ones_like(tokenized_input), output_hidden_states=True, output_attentions=True)  
                    attns_without_suffix = []

                    for (l, layername) in all_layers:
                        if "self_attn" in layername:
                            rep_layer = ret[layername].output
                            # print(rep_layer[1].shape)
                            # A^{i,j}_{l} will be H x T x T
                            attns_without_suffix.append(rep_layer[1])

                with torch.no_grad(), TraceDict(model.model, [l[1] for l in all_layers]) as ret:
                    # print(f"Ret keys: {ret.keys()}")
                    op_with_ntk = model.model(tokenized_attack_input, torch.ones_like(tokenized_attack_input), output_hidden_states=True, output_attentions=True)  
                    attns_with_suffix = []

                    for (l, layername) in all_layers:
                        if "self_attn" in layername:
                            rep_layer = ret[layername].output
                            # print(rep_layer[1].shape)
                            # A^{i,j}_{l} will be H x T x T
                            attns_with_suffix.append(rep_layer[1])

                
                tokens_without_suffix = [tokenizer.decode(t) for t in tokenized_input[0]]
                tokens_with_suffix = [tokenizer.decode(t) for t in tokenized_attack_input[0]]

                # print(f"Tokens with suffix: {tokens_with_suffix}")
            
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

                # attns_without_suffix = [attns_without_suffix_layer.sum(dim=0) for attns_without_suffix_layer in attns_without_suffix]
                # attns_with_suffix = [attns_with_suffix_layer.sum(dim=0) for attns_with_suffix_layer in attns_with_suffix]

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
                heatmap_diff = heatmap_diff.transpose(1, 0)

                # Switch the axes
                heatmap_without_suffix = heatmap_without_suffix.transpose(1, 0)
                heatmap_with_suffix = heatmap_with_suffix.transpose(1, 0) 

                fig = plot_attention_flow(
                    heatmap_diff, 
                    tokens_with_suffix, 
                    savepdf=f"{output_dir}/{model_name}/sample_{sample_i}/token_{ntk}_diff.png",
                    # figsize=(3, 24),
                    # figsize=(24, 3),
                    figsize=(40, 10),
                    # topk_prefix_start=-86,
                    topk_prefix_start=0,
                    topk_prefix_end=heatmap_diff.shape[1])
                
                fig = plot_attention_flow(
                    abs(heatmap_diff), 
                    tokens_with_suffix, 
                    savepdf=f"{output_dir}/{model_name}/sample_{sample_i}/token_{ntk}_diff_abs.png",
                    # figsize=(3, 24),
                    # figsize=(24, 3),
                    figsize=(40, 10),
                    # topk_prefix_start=-86,
                    topk_prefix_start=0,
                    topk_prefix_end=heatmap_diff.shape[1])

                fig = plot_attention_flow(
                    heatmap_with_suffix, 
                    tokens_with_suffix, 
                    savepdf=f"{output_dir}/{model_name}/sample_{sample_i}/token_{ntk}_with_suffix",
                    # figsize=(3, 24),
                    # figsize=(24, 3),
                    figsize=(40, 10),
                    # topk_prefix_start=-86,
                    topk_prefix_start=0,
                    topk_prefix_end=heatmap_with_suffix.shape[1])
                
                fig = plot_attention_flow(
                    heatmap_without_suffix, 
                    tokens_without_suffix, 
                    savepdf=f"{output_dir}/{model_name}/sample_{sample_i}/token_{ntk}_without_suffix",
                    # figsize=(3, 24),
                    # figsize=(24, 3),
                    figsize=(40, 10),
                    # topk_prefix_start=-86,
                    topk_prefix_start=0,
                    topk_prefix_end=heatmap_without_suffix.shape[1])


                for l in range(num_layers):
                    sample_attns_with_suffix[l] = max(sample_attns_with_suffix[l], np.max(heatmap_with_suffix[l, ante_pos]))
                    sample_attns_without_suffix[l] = max(sample_attns_without_suffix[l], np.max(heatmap_without_suffix[l, ante_pos]))
                    sample_attns_diff[l] = max(sample_attns_diff[l], np.max(heatmap_diff[l, ante_pos]))

        print(f"Tokens with suffix: {tokens_with_suffix}")
        print(f"Tokens without suffix: {tokens_without_suffix}")

        d = {"Reasoner": [f"{model_id}", ""], "Att. Wt.": ["Without suffix", "With suffix"]}
        for l in range(num_layers):
            all_attns_with_suffix[l].append(sample_attns_with_suffix[l])
            all_attns_without_suffix[l].append(sample_attns_without_suffix[l])
            all_attns_diff[l].append(sample_attns_diff[l])

            print(f"{l}")

            print(f"All attns with suffix (mean: {sum(all_attns_with_suffix[l])/len(all_attns_with_suffix[l])}): {all_attns_with_suffix[l]}")
            print(f"All attns without suffix (mean: {sum(all_attns_without_suffix[l])/len(all_attns_without_suffix[l])}): {all_attns_without_suffix[l]}")
            print(f"All attns diff (mean: {sum(all_attns_diff[l])/len(all_attns_diff[l])}): {all_attns_diff[l]}")

            print("\n---------------------\n")

            d[f"{l}"] = [sum(all_attns_without_suffix[l])/len(all_attns_without_suffix[l]), sum(all_attns_with_suffix[l])/len(all_attns_with_suffix[l])]
        df = pd.DataFrame(d)

        print(df.to_latex(index=False, float_format="{:.2f}".format,))
