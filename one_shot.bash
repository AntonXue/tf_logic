### One-shot experiments


TRAIN_LEN=100
EVAL_LEN=100
NUM_EPOCHS=25

python3 experiments/synthetic_experiments.py \
    --syn_exp_name one_shot \
    --model_name gpt2 \
    --num_rules 20 \
    --num_vars 5 \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS

python3 experiments/synthetic_experiments.py \
    --syn_exp_name one_shot \
    --model_name gpt2 \
    --num_rules 20 \
    --num_vars 10 \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS

python3 experiments/synthetic_experiments.py \
    --syn_exp_name one_shot \
    --model_name gpt2 \
    --num_rules 20 \
    --num_vars 15 \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS

python3 experiments/synthetic_experiments.py \
    --syn_exp_name one_shot \
    --model_name gpt2 \
    --num_rules 20 \
    --num_vars 20 \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS

python3 experiments/synthetic_experiments.py \
    --syn_exp_name one_shot \
    --model_name gpt2 \
    --num_rules 20 \
    --num_vars 25 \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS


