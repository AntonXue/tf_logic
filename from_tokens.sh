### One-shot experiments


TRAIN_LEN=16384
EVAL_LEN=2048
NUM_EPOCHS=64

EMBED_DIM=1024
NUM_LAYERS=2
NUM_HEADS=4

MIN_NUM_RULES=10
MAX_NUM_RULES=40
MIN_NUM_STATES=5
MAX_NUM_STATES=10

TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=128

MODEL_NAME="gpt2"

ANTE_PROB=0.3
CONSEQ_PROB=0.3
STATE_PROB=0.3


python3 experiments/synthetic_experiments.py \
    --syn_exp_name next_state_from_tokens \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --min_num_rules $MIN_NUM_RULES \
    --max_num_rules $MAX_NUM_RULES \
    --min_num_states $MIN_NUM_STATES \
    --max_num_states $MAX_NUM_STATES \
    --num_vars 10 \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob $STATE_PROB \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name next_state_from_tokens \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --min_num_rules $MIN_NUM_RULES \
    --max_num_rules $MAX_NUM_RULES \
    --min_num_states $MIN_NUM_STATES \
    --max_num_states $MAX_NUM_STATES \
    --num_vars 20 \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob $STATE_PROB \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name next_state_from_tokens \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --min_num_rules $MIN_NUM_RULES \
    --max_num_rules $MAX_NUM_RULES \
    --min_num_states $MIN_NUM_STATES \
    --max_num_states $MAX_NUM_STATES \
    --num_vars 30 \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob $STATE_PROB \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name next_state_from_tokens \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --min_num_rules $MIN_NUM_RULES \
    --max_num_rules $MAX_NUM_RULES \
    --min_num_states $MIN_NUM_STATES \
    --max_num_states $MAX_NUM_STATES \
    --num_vars 40 \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob $STATE_PROB \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments.py \
    --syn_exp_name next_state_from_tokens \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --min_num_rules $MIN_NUM_RULES \
    --max_num_rules $MAX_NUM_RULES \
    --min_num_states $MIN_NUM_STATES \
    --max_num_states $MAX_NUM_STATES \
    --num_vars 50 \
    --ante_prob $ANTE_PROB \
    --conseq_prob $CONSEQ_PROB \
    --state_prob $STATE_PROB \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE


