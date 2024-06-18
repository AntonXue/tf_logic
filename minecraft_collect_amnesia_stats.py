import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from minecraft_attacks_utils import (
    load_next_state_model_from_wandb,
    generate_text,
    generation_matches_label,
    minecraftexp_args_to_wandb_run_name,
    get_param_dict_list_from_config_file,
)
from argparse import ArgumentParser
import itertools


def run_check(prompt, suffix, label, tokenizer, model, device="cuda"):
    model.to(device)
    attack_prompt = f"{prompt}{suffix}"
    gen = generate_text(model, tokenizer, attack_prompt, device=device).split(
        "I cannot"
    )[0]

    print(f"Target: {label}")
    print(f"Gen: {gen}")
    return generation_matches_label(label, gen)


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
        "--amnesia_rule_depth",
        type=int,
        default=-1,
        help="Depth of the rules to remove for amnesia (if multiple rules as depth, one with randomly be chosen). Default (-1) is the general suppress rule attack",
    )

    args = parser.parse_args()
    config_file = args.config_file

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

            if args.amnesia_rule_depth >= 0:
                exp_id = exp_id + f"_amnesia_depth{args.amnesia_rule_depth}"
            else:
                raise ValueError("Amnesia rule depth must be >= 0")

            # # exp_id = "MinecraftNTP_gpt2_pt___eftext_msl768_ns1_nv1-64_mndt8_tbs32_ebs32_ntr65536_ntt16392_seed201"
            # # amnesia_depth = 1
            # amnesia_depth = 2
            # # exp_id = "MinecraftNTP_LMEval_gpt2_pt___eftext_msl768_ns3_nv1-64_mndt8_tbs32_ebs32_ntr65536_ntt16392_seed201"
            # exp_id = "MinecraftNTP_LMEval_gpt2_pt___eftext_msl768_ns5_nv1-64_mndt8_tbs32_ebs32_ntr65536_ntt16392_seed201"

            file_path = f"{args.output_dir}/{exp_id}_amnesia_depth{args.amnesia_rule_depth}.json"
            data = json.load(open(file_path))

            overlap_arr = []
            success_before = 0
            success_after = 0
            num_samples = 0

            for i in range(len(data)):
                found_sample = False
                for j, sample in enumerate(data[i]["train_sample_results"][-1::-1]):
                    if sample["attack_success"] and sample["qed"]:
                        found_sample = True
                        break
                if not found_sample:
                    continue
                num_samples += 1
                recipe_suppressed = sample["recipe_suppressed"]
                recipe_str = ""
                for r in recipe_suppressed:
                    recipe_str += f" {r[1].replace('minecraft:', '').replace('_', ' ')}"
                print(recipe_suppressed)
                print(recipe_str)
                attack_target = sample["attack_target"]
                attack_prompt = sample["original_prompt"]
                adv_suffix = sample["adversarial_suffix"]

                tok_atk_tgt = [
                    tokenizer.decode(i) for i in tokenizer(attack_target).input_ids
                ]
                tok_atk_prompt = [
                    tokenizer.decode(i) for i in tokenizer(attack_prompt).input_ids
                ]
                tok_adv_suf = [
                    tokenizer.decode(i) for i in tokenizer(adv_suffix).input_ids
                ]
                tok_recipe_str = [
                    tokenizer.decode(i) for i in tokenizer(recipe_str).input_ids
                ]

                print(f"Attack target: {attack_target}")
                print(f"Recipe str: {recipe_str}")
                print(f"Tok recipe str: {tok_recipe_str}")

                print(f"Adv suff: {adv_suffix}")
                print(f"Tok adv suff: {tok_adv_suf}")

                intersection = []
                tok_atk_tgt_no_prompt = []

                for t in tok_recipe_str:
                    tok_atk_tgt_no_prompt.append(t)
                    for t_p in tok_adv_suf:
                        if t.strip() in t_p.strip():
                            intersection.append((t, t_p))
                            break

                intersection_toks = [t[1] for t in intersection]

                tok_atk_suff_replaced = [
                    t if t not in intersection_toks else " and" for t in tok_adv_suf
                ]

                print(f"Tokenized target without common toks: {tok_atk_tgt_no_prompt}")

                print(f"Intersection: {intersection}")
                print(f"Replaced toks in suff: {tok_atk_suff_replaced}")
                print(f"% Overlap: {len(intersection) / len(tok_atk_tgt_no_prompt)}")

                orig_prompt = sample["original_prompt"]
                attack_target = attack_target.split("I cannot")[0]

                print("\n============================\n")
                before = run_check(
                    orig_prompt,
                    adv_suffix,
                    attack_target,
                    tokenizer,
                    model,
                    device=args.device,
                )
                print(f"Before replace: {before}")
                print("\n============================\n")
                after = run_check(
                    orig_prompt,
                    "".join(tok_atk_suff_replaced),
                    attack_target,
                    tokenizer,
                    model,
                    device=args.device,
                )
                print(f"After replace: {after}")

                success_before += before
                success_after += after

                overlap_arr.append(len(intersection) / len(tok_atk_tgt_no_prompt))
                # print(f"% Overlap: {len(intersection) / len(tok_atk_tgt)}")
                # print(f"Intersection: {set(tok_atk_tgt).intersection(set(tok_atk_prompt))}")
                print("\n++++++++++++++++++++++++\n")

            overlap_arr = np.array(overlap_arr)

            print(f"Num samples: {num_samples}")
            print(f"Mean overlap percentage: {np.mean(overlap_arr)}")
            print(f"Std overlap percentage: {np.std(overlap_arr)}")
            print(f"Max overlap percentage: {np.max(overlap_arr)}")
            print(f"Min overlap percentage: {np.min(overlap_arr)}")
            print(f"Success before: {success_before} | {success_before/num_samples}")
            print(f"Success after: {success_after} | {success_after/num_samples}")
