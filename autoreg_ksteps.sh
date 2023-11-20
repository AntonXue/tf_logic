### One-shot experiments


TRAIN_LEN=10000
EVAL_LEN=1000
NUM_EPOCHS=20
NUM_STEPS=10

EMBED_DIM=360

TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16

python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name gpt2 \
    --embed_dim $EMBED_DIM \
    --num_rules 20 \
    --num_vars 5 \
    --num_steps $NUM_STEPS \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE


python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name gpt2 \
    --embed_dim $EMBED_DIM \
    --num_rules 20 \
    --num_vars 10 \
    --num_steps $NUM_STEPS \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name gpt2 \
    --embed_dim $EMBED_DIM \
    --num_rules 20 \
    --num_vars 15 \
    --num_steps $NUM_STEPS \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name gpt2 \
    --embed_dim $EMBED_DIM \
    --num_rules 20 \
    --num_vars 20 \
    --num_steps $NUM_STEPS \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name gpt2 \
    --embed_dim $EMBED_DIM \
    --num_rules 20 \
    --num_vars 25 \
    --num_steps $NUM_STEPS \
    --ante_prob 0.2 \
    --conseq_prob 0.2 \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE


