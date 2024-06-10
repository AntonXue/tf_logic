# Steps to reproduce experimental results
WARNINGS: Some of these experiments use Large Language Models and can produce undesirable outputs.

## Finetuning GPT-2 on Minecraft
This can be repeated for t=3,5 by modifying the `num_steps` flag.

```
python experiments/minecraft_experiments_base.py 
    --model_name gpt2 
    --syn_exp_name autoreg_ksteps 
    --example_format text 
    --max_sequence_length 768 
    --max_num_distractors_triggerable 8 
    --seed 201 
    --num_steps 1 
    --min_num_vars 1 
    --max_num_vars 64 
    --train_len 65536 
    --eval_len 16392 
    --num_epochs 25 
    --train_batch_size 32 
    --eval_batch_size 32
```

## Running attacks on the Minecraft dataset

### Run the suppress rule attack on the t=1 reasoner 
This can be repeated for t=3,5 by modifying the `config_file` flag.

```
python minecraft_attacks_eval_suppress_rule_universal.py \
    --config_file configs/minecraft/attack_1.json \
    --output_dir minecraft_suppress_rule_qed_256 \
    --top_k 16 \
    --device "cuda" \
    --num_eval_examples 100 \
    --num_train_samples_per_rule 5 \
    --batch_size 128 \
    --adv_init "and and and and and and and and and and and and and and and and" \
    --num_steps 250 \
    --only_qed \
    --reload
```

### Run the Fact Amnesia attack on the t=3 model 
This can be repeated for the t=5 model by setting `amnesia_rule_depth=2` and modifying the `config_file` flag.

```
python minecraft_attacks_eval_suppress_rule_universal.py \
    --config_file configs/minecraft/attack_3.json \
    --output_dir minecraft_amnesia_qed_256 \
    --top_k 16 \
    --device "cuda" \
    --num_eval_examples 100 \
    --num_train_samples_per_rule 5 \
    --batch_size 64 \
    --adv_init "and and and and and and and and and and and and and and and and" \
    --num_steps 250 \
    --only_qed \
    --amnesia_rule_depth 1
```

### Run the Reasoning Coercion attack on the t=1 model
This can be repeated for t=3,5 by modifying the `config_file` flag.

```
python minecraft_attacks_eval_coerce_universal.py \
    --config_file configs/minecraft/attack_1.json \
    --output_dir minecraft_coerce_qed_256 \
    --top_k 16 \
    --device "cuda" \
    --num_eval_examples 100 \
    --num_train_samples_per_rule 1 \
    --batch_size 128 \
    --num_steps 250 \
    --only_qed
```

## Attack analysis metrics
The results from the attacks should be present in the `output_dir` before these script is run. 
This can be repeated for t=3,5 by modifying the `config_file` flag.

### Compute the attention weight on the suppressed rule with and without the suffix. 

```
python minecraft_attack_plots_heatmap_suppression_universal_stats.py 
    --config_file configs/minecraft/attack_1.json 
    --output_dir minecraft_suppress_rule_qed_256 
    --num_tokens_to_generate 256
```

### Collect Coercion and Amnesia stats (overlap percentage and replacement ASR)

```
python minecraft_collect_coercion_stats.py
    --config_file configs/minecraft/attack_1.json 
    --output_dir minecraft_coerce_qed_256 
```

```
python minecraft_collect_amnesia_stats.py
    --config_file configs/minecraft/attack_1.json 
    --output_dir minecraft_amnesia_qed_256 
```

## Attention suppression heatmaps for Llama-2-7b-chat

```
python plot_llama_heatmaps.py
    --model_name "llama-2-7b-chat-hf"
    --dataset_json_path data/rule_following.json
    --output_dir attention_heatmaps_llama
```

