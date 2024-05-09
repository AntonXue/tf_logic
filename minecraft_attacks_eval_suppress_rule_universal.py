import sys
from pathlib import Path
PROJ_ROOT = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, PROJ_ROOT)

from argparse import ArgumentParser
from minecraft.dataset import MinecraftAutoregKStepsNVarsDataset, MinecraftAutoregKStepsNVarsSupressAttackDataset
import json
import itertools
import os
import numpy as np
import gc

from safetensors import safe_open
import wandb
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from llm_attacks.minimal_gcg.opt_utils import token_gradients, sample_control, get_logits, target_loss
from llm_attacks.minimal_gcg.opt_utils import get_filtered_cands
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from llm_attacks import get_nonascii_toks

from transformers import utils
from bertviz import model_view
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
    raise_exception_if_not_found: str = True
):
    if not artifact_dir.is_dir() or overwrite:
        if not quiet:
            print(f"Querying id: {artifact_id}")

        try:
            api = wandb.Api()
            artifact = api.artifact(str(Path(wandb_project, artifact_id)), type=artifact_type)

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
    artifact_dir = download_artifact(artifact_id=artifact_id, 
                                     artifact_dir=artifact_dir,
                                     wandb_project=wandb_project,
                                     quiet=quiet,
                                     overwrite=overwrite)

    tensors = None
    if "model.safetensors" in os.listdir(artifact_dir):
        artifact_file = Path(artifact_dir, "model.safetensors")

        with safe_open(str(artifact_file), framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
    elif "pytorch_model.bin" in os.listdir(artifact_dir):
        artifact_file = Path(artifact_dir, "pytorch_model.bin")
        tensors = torch.load(artifact_file, map_location="cpu")
    else:
        raise Exception(f"No model.safetensors or pytorch_model.bin found in {artifact_dir}")

    model.load_state_dict(tensors)
    model.eval()
    return model

def minecraftexp_args_to_wandb_run_name(param_instance):
    model_str = f"{param_instance['model_name']}" + \
                (f"_pt" if param_instance['use_pretrained'] else "")
    
    if param_instance["syn_exp_name"] == "autoreg_ksteps":
        # MinecraftNTP: Next Token Prediction
        # return f"MinecraftNTP_{model_str}__" + \
        return f"MinecraftNTP_LMEval_{model_str}__" + \
            f"_ef{param_instance['example_format']}" + \
            f"_msl{param_instance['max_sequence_length']}" + \
            f"_ns{param_instance['num_steps']}" + \
            f"_nv{param_instance['min_num_vars']}-{param_instance['max_num_vars']}" + \
            f"_mndt{param_instance['max_num_distractors_triggerable']}" + \
            f"_tbs{param_instance['train_batch_size']}_ebs{param_instance['eval_batch_size']}" + \
            f"_ntr{param_instance['train_len']}_ntt{param_instance['eval_len']}_seed{param_instance['seed']}"

    raise ValueError(f"Unsupported syn_exp_name: {param_instance['syn_exp_name']}")

def get_param_dict_list_from_config_file(config_file) -> list:
    """Returns a list of dictionaries of parameters from a JSON config file."""
    config = json.load(open(config_file, "r"))
    return config

def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    # if gen_config.max_new_tokens > 50:
    #     print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
        
    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)

    # if gen_config is None:
    #     gen_config = model.generation_config
    #     # gen_config.max_length = 1024
    #     gen_config.max_new_tokens = 1024 - input_ids.shape[1]

    # print(f"Input ids: {input_ids}")
    # print(f"Input ids shape: {input_ids.shape}")
    # print(f"pad token id: {tokenizer.pad_token_id}")

    attn_masks = torch.ones_like(input_ids).to(model.device)
    try:
        output_ids = model.generate(input_ids, 
                                    attention_mask=attn_masks, 
                                    # generation_config=gen_config,
                                    # max_length=1024,
                                    max_new_tokens=256,
                                    pad_token_id=tokenizer.pad_token_id)[0]
    except Exception as e:
        print(f"Caught exception: {e}")
        # print(f"Input ids: {input_ids.item()}")
        print(f"Input ids shape: {input_ids.shape}")
        print(f"Gen config max new tokens: {gen_config.max_new_tokens}")
        print(f"pad token id: {tokenizer.pad_token_id}")
        print(f"Prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
        exit()

    return output_ids[assistant_role_slice.stop:].detach().cpu()

def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, target, gen_config=None):
    gen_str = tokenizer.decode(generate(model, 
                                        tokenizer, 
                                        input_ids, 
                                        assistant_role_slice, 
                                        gen_config=gen_config)).strip()
    # jailbroken = not any([prefix in gen_str for prefix in test_prefixes])
    # print(f"Target: {target}")
    # print(f"Generated: {gen_str}")
    # jailbroken = target in gen_str
    jailbroken = generation_matches_label(target, gen_str)
    return jailbroken

def get_allowed_tokens(tokenizer, dataset):
    allowed_tokens = set()
    print(f"Length of dataset: {len(dataset)}")
    print(f"Length of dataset.dataset: {len(dataset.dataset)}")
    for i in range(len(dataset)):
        # print(recipes)
        recipes = dataset[i]
        for sample in recipes:
            input_ids = tokenizer(sample['data'], return_tensors="pt").input_ids
            allowed_tokens.update(input_ids.flatten().tolist())
    return allowed_tokens 

def run_gcg_attack(args, model, tokenizer, user_prompt, label, not_allowed_tokens=None):
    num_steps = args.num_steps
    batch_size = args.batch_size
    topk = args.top_k
    # allow_non_ascii = args.allow_non_ascii
    adv_string_init = args.adv_init
    target = args.adv_target
    device = args.device
    suffix_manager = SuffixManager(tokenizer=tokenizer, 
              conv_template=load_conversation_template("gpt2"), 
              instruction=user_prompt, 
              target=target, 
              adv_string=adv_string_init)
    
    adv_suffix = adv_string_init

    is_success = False
    current_loss = torch.tensor(float('inf')).to(device)
    best_new_adv_suffix = adv_suffix

    # for i in tqdm(range(num_steps), desc=f"\nPassed:{is_success}\nCurrent Loss:{current_loss.detach().cpu().numpy()}\nCurrent Suffix:{best_new_adv_suffix}"):
    # for step in range(num_steps):
    tns_bar = trange(num_steps)
    for step in tns_bar:
        # tns_bar.set_description(f"Passed:{is_success} | Loss:{current_loss.detach().cpu().numpy()} | Suffix:{best_new_adv_suffix}", refresh=True)
        # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix)

        # print(f"Size of input ids: {input_ids.shape}")
        input_ids = input_ids.to(device)


        # Resize the input_ids to 1024 and append 220 to match that length if it is less than 1024
        # if input_ids.shape[0] < 1024:
        #     input_ids = torch.cat([input_ids, torch.ones(1024 - input_ids.shape[0], dtype=torch.long, device=device) * 220], dim=0)

        
        # print(f"Size of input ids: {input_ids.shape}")

        # # If input_ids is longer than 1024, just terminate the attack
        if input_ids.shape[0] > 1024:
            print(f"Input ids length is greater than 1024: {input_ids.shape[0]}")
            break
        
        # Step 2. Compute Coordinate Gradient
        coordinate_grad = token_gradients(model, 
                        input_ids, 
                        suffix_manager._control_slice, 
                        suffix_manager._target_slice, 
                        suffix_manager._loss_slice)
        
        # Step 3. Sample a batch of new tokens based on the coordinate gradient.
        # Notice that we only need the one that minimizes the loss.
        with torch.no_grad():
            
            # Step 3.1 Slice the input to locate the adversarial suffix.
            adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)

            # Step 3.2 Randomly sample a batch of replacements.
            new_adv_suffix_toks = sample_control(adv_suffix_tokens, 
                        coordinate_grad, 
                        batch_size, 
                        topk=topk, 
                        temp=1, 
                        not_allowed_tokens=not_allowed_tokens)
            
            # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
            # This step is necessary because tokenizers are not invertible
            # so Encode(Decode(tokens)) may produce a different tokenization.
            # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
            new_adv_suffix = get_filtered_cands(tokenizer, 
                                                new_adv_suffix_toks, 
                                                filter_cand=True,  # Setting this to True may reduce the number of candidates. 
                                                curr_control=adv_suffix)
            
            # exit()
            
            # Step 3.4 Compute loss on these candidates and take the argmin.
            logits, ids = get_logits(model=model, 
                                    tokenizer=tokenizer,
                                    input_ids=input_ids,
                                    control_slice=suffix_manager._control_slice, 
                                    test_controls=new_adv_suffix, 
                                    return_ids=True,
                                    batch_size=256) # decrease this number if you run into OOM.

            losses = target_loss(logits, ids, suffix_manager._target_slice)

            best_new_adv_suffix_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

            # print(f"Best new adv suffix: {best_new_adv_suffix}")
            # # Check number of tokens in the best new adv suffix
            # print(f"Number of tokens in best new adv suffix: {len(tokenizer(best_new_adv_suffix, add_special_tokens=False).input_ids)}")

            current_loss = losses[best_new_adv_suffix_id]

            # Update the running adv_suffix with the best candidate
            adv_suffix = best_new_adv_suffix
            is_success = check_for_attack_success(model, 
                                    tokenizer,
                                    suffix_manager.get_input_ids(adv_string=adv_suffix).to(device), 
                                    suffix_manager._control_slice, 
                                    target)
            
            # Check if the original conclusion is not present in the generated string
            # is_success = is_success and not check_for_attack_success(model,
            #                                                             tokenizer,
            #                                                             suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
            #                                                             suffix_manager._control_slice,
            #                                                             target.replace("cannot", "can"))
            

        tns_bar.set_description(f"Passed:{is_success} | Loss:{current_loss.detach().cpu().numpy():.3f} | Inp size:{input_ids.shape[0]}", refresh=True)
        # print(f"\nPassed:{is_success}\nCurrent Loss:{current_loss.detach().cpu().numpy()}\nCurrent Suffix:{best_new_adv_suffix}", end='\r')

        if is_success:
            break
        # if strict_is_success:
        #     break

        # (Optional) Clean up the cache.
        del coordinate_grad, adv_suffix_tokens ; gc.collect()
        torch.cuda.empty_cache()

    # input_ids = suffix_manager.get_input_ids(adv_string=best_new_adv_suffix).to(device)
    # gen_config = model.generation_config
    # gen_config.max_length = 1024
        
    # is_success = strict_is_success

    return is_success, {
        "original_prompt": user_prompt,
        "label": label,
        "num_steps": step+1,
        "loss": current_loss.detach().cpu().numpy().item(),
        "adversarial_suffix": best_new_adv_suffix,
        "adv_string_init": adv_string_init,
        "target": target,
        "batch_size": batch_size,
        "topk": topk,
        # "allow_non_ascii": allow_non_ascii,
    }

def generation_matches_label(label, generation):
    all_targets_in_labels = [t.split(".")[0].strip() for t in label.split("so I can create ")[1:]]
    all_targets_in_labels = set(["so I can create " + target for target in all_targets_in_labels])
    all_targets_in_generation = [t.split(".")[0].strip() for t in generation.split("so I can create ")[1:]]
    all_targets_in_generation = set(["so I can create " + target for target in all_targets_in_generation])

    # Check if the two lists are equal
    return all_targets_in_labels == all_targets_in_generation

def get_samples_from_dataset(eval_dataset, num_samples, target_depth, model, tokenizer, device="cuda", prompts=[]):
    samples = []
    example_idx = 0
    while len(samples) < num_samples and example_idx < len(eval_dataset):
        recipes = eval_dataset[example_idx]
        recipe = recipes[0]
        example_idx += 1
        # Only consider samples with the reasoning depth equal to the number of steps
        if recipe['depth'] < target_depth:
            continue
        prompt, label = get_prompt_and_label(recipe['data'])
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
    prompt = '\n'.join(sample_data.split('\n')[:5])
    label = '\n'.join(sample_data.split('\n')[5:])
    return prompt, label

def generate_text(model, tokenizer, prompt, max_length=1024, device="cuda", default_response="<PROMPT_TOO_LONG>"):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] > model.config.max_position_embeddings:
        return default_response
    # Stop generating when "I cannot create any other items." is generated
    # generation = model.generate(input_ids, max_length=max_length)
    generation = model.generate(input_ids, max_new_tokens=256)
    generation = tokenizer.decode(generation[0], skip_special_tokens=True)
    return generation[len(prompt):]

def get_rule_suppressed_adv_target(generation, recipe_to_suppress):
    consequents_to_remove = [r[1].replace('minecraft:', '').replace('_', ' ') for r in recipe_to_suppress]
    adv_target = ''
    # Remove all lines in adv_target which have 'so I can create consequent' for all consequents in the recipe to remove
    for line in generation.split('.'):
        if any([f'so I can create {consequent}' in line for consequent in consequents_to_remove]):
            continue
        if len(line.strip()) > 0:
            adv_target += line + '.'
    adv_target = adv_target.strip()
    # adv_target += "\nI cannot create any other items."
    # Return adv_target and the recipe to suppress
    return adv_target, [(list(antecedents), consequent, depth) for (antecedents, consequent, depth) in recipe_to_suppress]

def log_final_metrics(output_dir, exp_id, num_successes, num_total, accuracy):
    with open(os.path.join(output_dir, f"{exp_id}_final_{time.time()}.json"), "w") as f:
        metrics = {
            "num_successes": num_successes,
            "num_total": num_total,
            "accuracy": accuracy
        }
        json.dump(metrics, f)
    print(f"Successfully logged final metrics to {output_dir}/{exp_id}_final_{time.time()}.json")

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
            print(f"Length of model embeddings: {model.get_input_embeddings().weight.shape[0]}")
            model.resize_token_embeddings(len(tokenizer))
            print(f"Length of model embeddings after resizing: {model.get_input_embeddings().weight.shape[0]}")
            print(f"Model vocab size: {model.config.vocab_size}")

            eval_dataset = MinecraftAutoregKStepsNVarsSupressAttackDataset(
                                num_steps=param_instance["num_steps"],
                                num_vars_range=(param_instance["min_num_vars"], param_instance["max_num_vars"]),
                                max_num_distractors_triggerable=param_instance.get("max_num_distractors_triggerable", 3),
                                example_format=param_instance.get("example_format", "text"),
                                dataset_len=max(param_instance.get("eval_len"), args.num_eval_examples),
                                seed=param_instance["seed"],
                                tokenizer=tokenizer,
                                task_type="next_token_prediction",
                                num_samples_per_recipe=args.num_train_samples_per_rule,
                                only_qed=args.only_qed,
                            )
            
            not_allowed_tokens = None
            if not args.allow_tokens_not_in_dataset:
                all_tokens_in_vocab = set(range(tokenizer.vocab_size))
                tokens_in_dataset = get_allowed_tokens(tokenizer, eval_dataset)
                not_allowed_tokens = torch.tensor(list(all_tokens_in_vocab - tokens_in_dataset), device=args.device)
                print(f"Number of tokens in dataset: {len(tokens_in_dataset)}")
            elif not args.allow_non_ascii:
                not_allowed_tokens = get_nonascii_toks(tokenizer)
                print(f"Number of non-ascii tokens: {len(not_allowed_tokens)}")

            samples, prompts = get_samples_from_dataset(eval_dataset, args.num_eval_examples, param_instance["num_steps"] - 1, model, tokenizer, device=args.device)
            results = []
                
            for epoch in range(args.num_epochs):
                num_successes = 0
                acc = 0
                # Maximum number of retries if CUDA runs out of memory
                max_num_tries = 5

                # print(f"Running attack for {exp_id} | Start idx: {start_idx} | Num total: {num_total} | Accuracy: {acc}")
                ted_bar = trange(0, len(samples), desc=f"Epoch: {epoch} | Running accuracy: {acc}")
                for eval_i in ted_bar:
                    recipes = samples[eval_i]
                    num_tries = 0
                    # args.num_steps = total_num_steps
                    args.adv_init = original_adv_init

                    # Get adv_init from last epoch's results
                    if len(results) > 0:
                        last_epoch_results = [r for r in results if r["eval_idx"] == eval_i and r["epoch"] == epoch - 1]
                        if len(last_epoch_results) > 0:
                            args.adv_init = last_epoch_results[0]["adversarial_suffix"]

                    epoch_adv_init = args.adv_init


                    ted_bar.set_description(f"Epoch: {epoch} | Running accuracy: {acc} | Example Idx: {eval_i}", refresh=True)
                    
                    idx_results = []

                    for recipe in recipes:
                        prompt, label = get_prompt_and_label(recipe['data'])
                        generation = generate_text(model, tokenizer, prompt, device=args.device)
                        args.adv_target, recipe_suppressed = get_rule_suppressed_adv_target(generation, recipe['recipe_from_idx'])
                        generation = generation.split("I cannot create any other items.")[0]
                        
                        attack_result = {}
                        success = False
                        num_tries = 0
                        attack_start_time = time.time()

                        while num_tries < max_num_tries:
                            try:
                                success, attack_result = run_gcg_attack(args, model, tokenizer, prompt, label, not_allowed_tokens)
                                break
                            except RuntimeError as e:
                                log_final_metrics(args.output_dir, exp_id, num_successes, len(results), acc)
                                print(f"Caught exception: {e} | Trying again in 5 seconds...")
                                torch.cuda.empty_cache()
                                time.sleep(5)
                                num_tries += 1
                                continue

                        if "cuda" in args.device:
                            torch.cuda.synchronize(device=args.device)
                        attack_end_time = time.time()

                        attack_result["original_generation"] = generation
                        attack_result["original_generation_matches_label"] = generation_matches_label(label, attack_result["original_generation"])
                        attack_result["attack_target"] = args.adv_target
                        attack_result["target_in_original_generation"] = (args.adv_target == attack_result["original_generation"])
                        attack_result["attack_time"] = attack_end_time - attack_start_time

                        attack_result["attack_prompt"] = f"{prompt}{attack_result['adversarial_suffix']}"
                        attack_result["attack_generation"] = generate_text(model, tokenizer, attack_result["attack_prompt"], device=args.device)
                        attack_result["attack_generation_matches_label"] = generation_matches_label(label, attack_result["attack_generation"])
                        attack_result["attack_success"] = success
                        attack_result["recipe_suppressed"] = recipe_suppressed
                        attack_result["qed"] = recipe["labels"].item()
                        idx_results.append(attack_result)

                        # update adv_init if the attack was successful
                        if success:
                            args.adv_init = attack_result["adversarial_suffix"]

                        # if args.num_steps <= 0:
                        #     break
                    # attack_end_time = time.time()

                    if sum([r["attack_success"] for r in idx_results]) == len(idx_results):
                        num_successes += 1
                        
                    acc = num_successes / (eval_i + 1)

                    final_idx_results = {
                        "eval_idx": eval_i,
                        "epoch": epoch,
                        "adv_init": epoch_adv_init,
                        "adversarial_suffix": args.adv_init,
                        "attack_success": success and len(idx_results) == len(recipes),
                        "attack_success_percentage": sum([r["attack_success"] for r in idx_results]) / len(idx_results),
                        "recipe_suppressed": [(list(antecedents), consequent, depth) for (antecedents, consequent, depth) in recipe["recipe_from_idx"]],
                        # "num_steps": total_num_steps - args.num_steps,
                        "num_steps": sum([r["num_steps"] for r in idx_results]),
                        "total_attack_time": sum([r["attack_time"] for r in idx_results]),
                        "train_sample_results": idx_results,
                        "train_running_accuracy": acc,
                    }

                    results.append(final_idx_results)

                    # Update the json file with the attack result
                    with open(os.path.join(args.output_dir, f"{exp_id}.json"), "w") as f:
                        json.dump(results, f)

            log_final_metrics(args.output_dir, exp_id, num_successes, len(results), acc)