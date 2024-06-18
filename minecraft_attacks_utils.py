import sys
from pathlib import Path

PROJ_ROOT = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, PROJ_ROOT)

import json
import os
import numpy as np
import gc

from safetensors import safe_open
import wandb
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from llm_attacks.minimal_gcg.opt_utils import (
    token_gradients,
    sample_control,
    get_logits,
    target_loss,
)
from llm_attacks.minimal_gcg.opt_utils import get_filtered_cands
from llm_attacks.minimal_gcg.string_utils import (
    SuffixManager,
    load_conversation_template,
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


def download_artifact(
    artifact_id: str,
    artifact_dir: str,
    wandb_project: str = "transformer_friends/transformer_friends",
    quiet: bool = False,
    overwrite: bool = False,
    artifact_type: str = "model",
    raise_exception_if_not_found: str = True,
):
    if not artifact_dir.is_dir() or overwrite:
        if not quiet:
            print(f"Querying id: {artifact_id}")

        try:
            api = wandb.Api()
            artifact = api.artifact(
                str(Path(wandb_project, artifact_id)), type=artifact_type
            )

            if not quiet:
                print(f"Downloading: {artifact}")
            artifact_dir = artifact.download(artifact_dir)
        except Exception as e:
            if raise_exception_if_not_found:
                raise Exception(e)
            artifact_dir = None
    return artifact_dir


def load_next_state_model_from_wandb(
    artifact_id: str,
    model: nn.Module,
    wandb_project: str = "transformer_friends/transformer_friends",
    quiet: bool = False,
    overwrite: bool = False,
):
    artifact_dir = Path("minecraft_dump", "artifacts", artifact_id)
    artifact_dir = download_artifact(
        artifact_id=artifact_id,
        artifact_dir=artifact_dir,
        wandb_project=wandb_project,
        quiet=quiet,
        overwrite=overwrite,
    )

    print(f"Artifact ID: {artifact_id}")

    tensors = None
    if "model.safetensors" in os.listdir(artifact_dir):
        print(f"Loading model from safetensors")
        artifact_file = Path(artifact_dir, "model.safetensors")

        with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
    elif "pytorch_model.bin" in os.listdir(artifact_dir):
        print(f"Loading model from pytorch_model.bin")
        artifact_file = Path(artifact_dir, "pytorch_model.bin")
        tensors = torch.load(artifact_file, map_location="cpu")
    else:
        raise Exception(
            f"No model.safetensors or pytorch_model.bin found in {artifact_dir}"
        )

    model.load_state_dict(tensors, strict=False)
    model.eval()
    return model


def minecraftexp_args_to_wandb_run_name(param_instance):
    model_str = f"{param_instance['model_name']}" + (
        f"_pt" if param_instance["use_pretrained"] else ""
    )

    if param_instance["syn_exp_name"] == "autoreg_ksteps":
        # Special condition for older 1-step experiments
        # Please remove this condition for all new experiments
        if param_instance["num_steps"] == 1:
            return (
                f"MinecraftNTP_{model_str}__"
                + f"_ef{param_instance['example_format']}"
                + f"_msl{param_instance['max_sequence_length']}"
                + f"_ns{param_instance['num_steps']}"
                + f"_nv{param_instance['min_num_vars']}-{param_instance['max_num_vars']}"
                + f"_mndt{param_instance['max_num_distractors_triggerable']}"
                + f"_tbs{param_instance['train_batch_size']}_ebs{param_instance['eval_batch_size']}"
                + f"_ntr{param_instance['train_len']}_ntt{param_instance['eval_len']}_seed{param_instance['seed']}"
            )

        return (
            f"MinecraftNTP_LMEval_{model_str}__"
            + f"_ef{param_instance['example_format']}"
            + f"_msl{param_instance['max_sequence_length']}"
            + f"_ns{param_instance['num_steps']}"
            + f"_nv{param_instance['min_num_vars']}-{param_instance['max_num_vars']}"
            + f"_mndt{param_instance['max_num_distractors_triggerable']}"
            + f"_tbs{param_instance['train_batch_size']}_ebs{param_instance['eval_batch_size']}"
            + f"_ntr{param_instance['train_len']}_ntt{param_instance['eval_len']}_seed{param_instance['seed']}"
        )

    raise ValueError(f"Unsupported syn_exp_name: {param_instance['syn_exp_name']}")


def get_param_dict_list_from_config_file(config_file) -> list:
    """Returns a list of dictionaries of parameters from a JSON config file."""
    config = json.load(open(config_file, "r"))
    return config


def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    input_ids = input_ids[: assistant_role_slice.stop].to(model.device).unsqueeze(0)

    attn_masks = torch.ones_like(input_ids).to(model.device)
    try:
        output_ids = model.generate(
            input_ids,
            attention_mask=attn_masks,
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
    except Exception as e:
        print(f"Caught exception: {e}")
        # print(f"Input ids: {input_ids.item()}")
        print(f"Input ids shape: {input_ids.shape}")
        print(f"Gen config max new tokens: {gen_config.max_new_tokens}")
        print(f"pad token id: {tokenizer.pad_token_id}")
        print(f"Prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
        exit()

    return output_ids[assistant_role_slice.stop :].detach().cpu()


def check_for_attack_success(
    model, tokenizer, input_ids, assistant_role_slice, target, gen_config=None
):
    gen_str = tokenizer.decode(
        generate(
            model, tokenizer, input_ids, assistant_role_slice, gen_config=gen_config
        )
    ).strip()
    # Uncomment the following line for a stricter match
    # jailbroken = generation_matches_label(target, gen_str)
    jailbroken = generation_matches_label(target, gen_str.split("I cannot")[0])
    return jailbroken


def get_allowed_tokens(tokenizer, dataset):
    allowed_tokens = set()
    print(f"Length of dataset: {len(dataset)}")
    print(f"Length of dataset.dataset: {len(dataset.dataset)}")
    for i in range(len(dataset)):
        # print(recipes)
        recipes = dataset[i]
        for sample in recipes:
            input_ids = tokenizer(sample["data"], return_tensors="pt").input_ids
            allowed_tokens.update(input_ids.flatten().tolist())
    return allowed_tokens


def run_gcg_attack(args, model, tokenizer, user_prompt, label, not_allowed_tokens=None):
    num_steps = args.num_steps
    batch_size = args.batch_size
    topk = args.top_k
    adv_string_init = args.adv_init
    target = args.adv_target
    device = args.device
    suffix_manager = SuffixManager(
        tokenizer=tokenizer,
        conv_template=load_conversation_template("gpt2"),
        instruction=user_prompt,
        target=target,
        adv_string=adv_string_init,
    )

    adv_suffix = adv_string_init

    is_success = False
    current_loss = torch.tensor(float("inf")).to(device)
    best_new_adv_suffix = adv_suffix

    tns_bar = trange(num_steps)
    for step in tns_bar:
        # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)

        input_ids = input_ids.to(device)

        # # If input_ids is longer than 1024, just terminate the attack
        if input_ids.shape[0] > 1024:
            print(f"Input ids length is greater than 1024: {input_ids.shape[0]}")
            break

        # Step 2. Compute Coordinate Gradient
        coordinate_grad = token_gradients(
            model,
            input_ids,
            suffix_manager._control_slice,
            suffix_manager._target_slice,
            suffix_manager._loss_slice,
        )

        # Step 3. Sample a batch of new tokens based on the coordinate gradient.
        # Notice that we only need the one that minimizes the loss.
        with torch.no_grad():
            # Step 3.1 Slice the input to locate the adversarial suffix.
            adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)

            # Step 3.2 Randomly sample a batch of replacements.
            new_adv_suffix_toks = sample_control(
                adv_suffix_tokens,
                coordinate_grad,
                batch_size,
                topk=topk,
                temp=1,
                not_allowed_tokens=not_allowed_tokens,
            )

            # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
            # This step is necessary because tokenizers are not invertible
            # so Encode(Decode(tokens)) may produce a different tokenization.
            # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
            new_adv_suffix = get_filtered_cands(
                tokenizer,
                new_adv_suffix_toks,
                filter_cand=True,  # Setting this to True may reduce the number of candidates.
                curr_control=adv_suffix,
            )

            # exit()

            # Step 3.4 Compute loss on these candidates and take the argmin.
            logits, ids = get_logits(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                control_slice=suffix_manager._control_slice,
                test_controls=new_adv_suffix,
                return_ids=True,
                batch_size=512,
            )  # decrease this number if you run into OOM.

            losses = target_loss(logits, ids, suffix_manager._target_slice)

            best_new_adv_suffix_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

            current_loss = losses[best_new_adv_suffix_id]

            # Update the running adv_suffix with the best candidate
            adv_suffix = best_new_adv_suffix
            is_success = check_for_attack_success(
                model,
                tokenizer,
                suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
                suffix_manager._control_slice,
                target,
            )

        tns_bar.set_description(
            f"Passed:{is_success} | Loss:{current_loss.detach().cpu().numpy():.3f} | Inp size:{input_ids.shape[0]}",
            refresh=True,
        )

        if is_success:
            break

        # (Optional) Clean up the cache.
        del coordinate_grad, adv_suffix_tokens
        gc.collect()
        torch.cuda.empty_cache()

    return is_success, {
        "original_prompt": user_prompt,
        "label": label,
        "num_steps": step + 1,
        "loss": current_loss.detach().cpu().numpy().item(),
        "adversarial_suffix": best_new_adv_suffix,
        "adv_string_init": adv_string_init,
        "target": target,
        "batch_size": batch_size,
        "topk": topk,
    }


def generation_matches_label(label, generation):
    all_targets_in_labels = [
        t.split(".")[0].strip() for t in label.split("so I can create ")[1:]
    ]
    all_targets_in_labels = set(
        ["so I can create " + target for target in all_targets_in_labels]
    )
    all_targets_in_generation = [
        t.split(".")[0].strip() for t in generation.split("so I can create ")[1:]
    ]
    all_targets_in_generation = set(
        ["so I can create " + target for target in all_targets_in_generation]
    )

    # Check if the two lists are equal
    return all_targets_in_labels == all_targets_in_generation


def serialize_recipe(recipe):
    # Sort recipe first by depth and then by the consequent
    recipe = sorted(recipe, key=lambda x: (x[2], x[1]))
    # Sort antecedents in each rule
    recipe = tuple(
        (tuple(sorted(antecedents)), consequent, depth)
        for (antecedents, consequent, depth) in recipe
    )
    return recipe


def get_samples_from_dataset(
    eval_dataset,
    num_samples,
    target_depth,
    model,
    tokenizer,
    device="cuda",
    prompts=[],
    example_idxs_to_ignore=[],
):
    samples = []
    example_idx = 0
    print(f"Number of idxs to ignore: {len(example_idxs_to_ignore)}")
    print(f"First 10 idxs to ignore: {example_idxs_to_ignore[:10]}")
    while len(samples) < num_samples and example_idx < len(eval_dataset):
        if example_idx in example_idxs_to_ignore:
            print(f"Example in examples to ignore... skipping...")
            example_idx += 1
            continue
        recipes = eval_dataset[example_idx]
        recipe = recipes[0]
        example_idx += 1
        # Only consider samples with the reasoning depth equal to the number of steps
        if recipe["depth"] < target_depth:
            continue
        prompt, label = get_prompt_and_label(recipe["data"])
        if prompt.strip() in prompts:
            continue
        generation = generate_text(model, tokenizer, prompt, device=device)
        # Only if the generation is correct, run the attack
        if generation_matches_label(label, generation):
            samples.append(recipes)
            prompts.append(prompt.strip())

    if example_idx >= len(eval_dataset) and len(samples) < num_samples:
        print("Ran out of examples to evaluate the attack on")

    return samples, prompts


def get_prompt_and_label(sample_data):
    prompt = "\n".join(sample_data.split("\n")[:5])
    label = "\n".join(sample_data.split("\n")[5:])
    return prompt, label


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=1024,
    device="cuda",
    default_response="<PROMPT_TOO_LONG>",
):
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    if tokenized_prompt.input_ids.shape[1] > model.config.max_position_embeddings:
        return default_response
    with torch.no_grad():
        generation = model.generate(**tokenized_prompt, max_new_tokens=256)
    generation = tokenizer.decode(generation[0], skip_special_tokens=True)
    return generation[len(prompt) :]


def generate_text_batch(
    model,
    tokenizer,
    prompts,
    max_length=1024,
    device="cuda",
    default_response="<PROMPT_TOO_LONG>",
    batch_size=1,
):
    input_ids = tokenizer(
        prompts, return_tensors="pt", padding="longest", truncation=True
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**input_ids, max_new_tokens=256)
    generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [generations[i][len(prompts[i]) :] for i in range(len(generations))]


def generation_not_contains_label(label, generation):
    all_targets_in_labels = [
        t.split(".")[0].strip() for t in label.split("so I can create ")[1:]
    ]
    all_targets_in_labels = set(
        ["so I can create " + target for target in all_targets_in_labels]
    )
    all_targets_in_generation = [
        t.split(".")[0].strip() for t in generation.split("so I can create ")[1:]
    ]
    all_targets_in_generation = set(
        ["so I can create " + target for target in all_targets_in_generation]
    )

    # Check if no target in the label is present in the generation
    return len(all_targets_in_labels.intersection(all_targets_in_generation)) == 0


def get_rule_suppressed_adv_target(
    generation, recipe_to_suppress, return_suppressed_string=False
):
    consequents_to_remove = [
        r[1].replace("minecraft:", "").replace("_", " ") for r in recipe_to_suppress
    ]
    adv_target = ""
    suppressed_string = ""
    # Remove all lines in adv_target which have 'so I can create consequent' for all consequents in the recipe to remove
    for line in generation.split("."):
        if any(
            [
                f"so I can create {consequent}" in line
                for consequent in consequents_to_remove
            ]
        ):
            suppressed_string += line + "."
            continue
        if len(line.strip()) > 0:
            adv_target += line + "."
    adv_target = adv_target.strip()

    recipe = [
        (list(antecedents), consequent, depth)
        for (antecedents, consequent, depth) in recipe_to_suppress
    ]

    if return_suppressed_string:
        return adv_target, recipe, suppressed_string
    # Return adv_target and the recipe to suppress
    return adv_target, recipe


def log_final_metrics(output_dir, exp_id, num_successes, num_total, accuracy):
    with open(os.path.join(output_dir, f"{exp_id}_final_{time.time()}.json"), "w") as f:
        metrics = {
            "num_successes": num_successes,
            "num_total": num_total,
            "accuracy": accuracy,
        }
        json.dump(metrics, f)
    print(
        f"Successfully logged final metrics to {output_dir}/{exp_id}_final_{time.time()}.json"
    )
