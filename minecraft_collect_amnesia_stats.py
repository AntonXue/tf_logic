import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from minecraft_attacks_eval_coerce_universal import load_next_state_model_from_wandb, generate_text, generation_matches_label

def run_check(prompt, suffix, label, tokenizer, model, device="cuda"):
    model.to(device)
    attack_prompt = f"{prompt}{suffix}"
    gen = generate_text(model, tokenizer, attack_prompt, device=device).split("I cannot")[0]

    print(f"Target: {label}")
    print(f"Gen: {gen}")
    return generation_matches_label(label, gen)

# exp_id = "MinecraftNTP_gpt2_pt___eftext_msl768_ns1_nv1-64_mndt8_tbs32_ebs32_ntr65536_ntt16392_seed201"
# amnesia_depth = 1
amnesia_depth = 2
# exp_id = "MinecraftNTP_LMEval_gpt2_pt___eftext_msl768_ns3_nv1-64_mndt8_tbs32_ebs32_ntr65536_ntt16392_seed201"
exp_id = "MinecraftNTP_LMEval_gpt2_pt___eftext_msl768_ns5_nv1-64_mndt8_tbs32_ebs32_ntr65536_ntt16392_seed201"
artifact_id = f"model-{exp_id}:v0"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = load_next_state_model_from_wandb(artifact_id, model)

file_path = f"minecraft_amnesia_qed_256/{exp_id}_amnesia_depth{amnesia_depth}.json"
# file_path = "minecraft_coerce_qed_256_long_suffix/MinecraftNTP_LMEval_gpt2_pt___eftext_msl768_ns3_nv1-64_mndt8_tbs32_ebs32_ntr65536_ntt16392_seed201.json"
# file_path = "minecraft_coerce_qed_256_long_suffix/MinecraftNTP_LMEval_gpt2_pt___eftext_msl768_ns5_nv1-64_mndt8_tbs32_ebs32_ntr65536_ntt16392_seed201.json"
data = json.load(open(file_path))

overlap_arr = []
success_before = 0
success_after = 0
num_samples = 0

for i in range(len(data)):
# for i in range(10):
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
        # for a in r[0]:
        #     recipe_str += f" {a.replace('minecraft:', '').replace('_', ' ')}"
        recipe_str += f" {r[1].replace('minecraft:', '').replace('_', ' ')}"
    print(recipe_suppressed)
    print(recipe_str)
    # exit()
    attack_target = sample["attack_target"]
    attack_prompt = sample["original_prompt"]
    adv_suffix = sample["adversarial_suffix"]

    tok_atk_tgt = [tokenizer.decode(i) for i in tokenizer(attack_target).input_ids]
    tok_atk_prompt = [tokenizer.decode(i) for i in tokenizer(attack_prompt).input_ids]
    tok_adv_suf = [tokenizer.decode(i) for i in tokenizer(adv_suffix).input_ids]
    tok_recipe_str = [tokenizer.decode(i) for i in tokenizer(recipe_str).input_ids]
    # print(f"Attack prompt: {attack_prompt}")
    # print(f"Tokenized attack prompt: {tok_atk_prompt}")

    # print("\n++++++++++++++++++++++++\n")

    print(f"Attack target: {attack_target}")
    print(f"Recipe str: {recipe_str}")
    print(f"Tok recipe str: {tok_recipe_str}")

    print(f"Adv suff: {adv_suffix}")
    print(f"Tok adv suff: {tok_adv_suf}")

    # print(f"Tokenized attack target: {tok_atk_tgt}")

    # # ['I', ' have', ' stripped', ' jungle', ' log', ' and', ' so', ' I', ' can', ' create', ' prism', 'arine', ' crystals', '.']
    # print("\n++++++++++++++++++++++++\n")

    intersection = []
    tok_atk_tgt_no_prompt = []

    for t in tok_recipe_str:
        # if t in tok_atk_prompt:
        #     print(f"{t} in tok atk prompt")
        # # if t in tok_atk_prompt:
        #     continue
        tok_atk_tgt_no_prompt.append(t)
        for t_p in tok_adv_suf:
            # if t.strip() in t_p.strip():
            if t.strip() in t_p.strip():
                intersection.append((t, t_p))
                break

    intersection_toks = [t[1] for t in intersection]

    tok_atk_suff_replaced = [t if t not in intersection_toks else " and" for t in tok_adv_suf]

    print(f"Tokenized target without common toks: {tok_atk_tgt_no_prompt}")

    print(f"Intersection: {intersection}")
    print(f"Replaced toks in suff: {tok_atk_suff_replaced}")
    print(f"% Overlap: {len(intersection) / len(tok_atk_tgt_no_prompt)}")

    orig_prompt = sample['original_prompt']
    attack_target = attack_target.split("I cannot")[0]

    print("\n============================\n")
    before = run_check(orig_prompt, adv_suffix, attack_target, tokenizer, model)
    print(f"Before replace: {before}")
    print("\n============================\n")
    after = run_check(orig_prompt, ''.join(tok_atk_suff_replaced), attack_target, tokenizer, model)
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


