### One-shot experiments


TRAIN_LEN=20000
EVAL_LEN=1000
NUM_EPOCHS=20
NUM_STEPS=4

NUM_RULES=20

EMBED_DIM=768
NUM_LAYERS=4
NUM_HEADS=4

TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32

MODEL_NAME="gpt2"

ANTE_PROB=0.2
CONSEQ_PROB=0.3

: '
python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --num_rules $NUM_RULES \
    --num_vars 5 \
    --num_steps $NUM_STEPS \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE
'

python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --num_rules $NUM_RULES \
    --num_vars 10 \
    --num_steps $NUM_STEPS \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --num_rules $NUM_RULES \
    --num_vars 15 \
    --num_steps $NUM_STEPS \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --num_rules $NUM_RULES \
    --num_vars 20 \
    --num_steps $NUM_STEPS \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name autoreg_ksteps \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --num_rules $NUM_RULES \
    --num_vars 25 \
    --num_steps $NUM_STEPS \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob 0.3 \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

