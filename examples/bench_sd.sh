MODEL_PATH=Qwen/Qwen2.5-7B
SPEC_MODEL_PATH=mit-han-lab/Qwen2.5-7B-Eagle-RL
DATA_PATH=data/Eurus_sample.json

SPEC_STEPS=8
NUM_DRAFT_TOKENS=48
TOPKS=4
MAX_BS=1
TP=2

python scripts/bench_speculative_decoding.py \
    --data_dir $DATA_PATH \
    --spec_algorithm EAGLE \
    --model_path $MODEL_PATH \
    --eagle_path $SPEC_MODEL_PATH \
    --speculative_num_steps $SPEC_STEPS \
    --speculative_eagle_topk $TOPKS \
    --speculative_num_draft_tokens $NUM_DRAFT_TOKENS \
    --tp_size $TP \
    --max_bs $MAX_BS \
    --attention_backend fa3