MODEL_NAME=$1
EMBED_DIM=$2
NUM_LAYERS=$3
NUM_HEADS=$4
NUM_VARS=$5

TRAIN_LEN=32768
EVAL_LEN=4096
NUM_EPOCHS=50

MIN_NUM_RULES=16
MAX_NUM_RULES=64
MIN_NUM_STATES=8
MAX_NUM_STATES=32

MIN_ANTE_PROB=0.3
MAX_ANTE_PROB=0.5
MIN_CONSEQ_PROB=0.2
MAX_CONSEQ_PROB=0.3
MIN_STATE_PROB=0.2
MAX_STATE_PROB=0.5

TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=128

python3 experiments/synthetic_experiments_base.py \
    --syn_exp_name next_state_from_tokens \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --min_num_rules $MIN_NUM_RULES \
    --max_num_rules $MAX_NUM_RULES \
    --min_num_states $MIN_NUM_STATES \
    --max_num_states $MAX_NUM_STATES \
    --num_vars $NUM_VARS \
    --min_ante_prob $MIN_ANTE_PROB \
    --max_ante_prob $MAX_ANTE_PROB \
    --min_conseq_prob $MIN_CONSEQ_PROB \
    --max_conseq_prob $MAX_CONSEQ_PROB \
    --min_state_prob $MIN_STATE_PROB \
    --max_state_prob $MAX_STATE_PROB \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

<< ////
python3 experiments/synthetic_experiments_base.py \
    --syn_exp_name next_state_from_tokens \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --min_num_rules $MIN_NUM_RULES \
    --max_num_rules $MAX_NUM_RULES \
    --min_num_states $MIN_NUM_STATES \
    --max_num_states $MAX_NUM_STATES \
    --num_vars 32 \
    --min_ante_prob $MIN_ANTE_PROB \
    --max_ante_prob $MAX_ANTE_PROB \
    --min_conseq_prob $MIN_CONSEQ_PROB \
    --max_conseq_prob $MAX_CONSEQ_PROB \
    --min_state_prob $MIN_STATE_PROB \
    --max_state_prob $MAX_STATE_PROB \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE

python3 experiments/synthetic_experiments_base.py \
    --syn_exp_name next_state_from_tokens \
    --model_name $MODEL_NAME \
    --embed_dim $EMBED_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --min_num_rules $MIN_NUM_RULES \
    --max_num_rules $MAX_NUM_RULES \
    --min_num_states $MIN_NUM_STATES \
    --max_num_states $MAX_NUM_STATES \
    --num_vars 64 \
    --min_ante_prob $MIN_ANTE_PROB \
    --max_ante_prob $MAX_ANTE_PROB \
    --min_conseq_prob $MIN_CONSEQ_PROB \
    --max_conseq_prob $MAX_CONSEQ_PROB \
    --min_state_prob $MIN_STATE_PROB \
    --max_state_prob $MAX_STATE_PROB \
    --train_len $TRAIN_LEN \
    --eval_len $EVAL_LEN \
    --num_epochs $NUM_EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE
////

