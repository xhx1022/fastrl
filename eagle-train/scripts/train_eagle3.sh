export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PROJECT_NAME=FastRL
EXPERIMENT_NAME=Eagle-Train
EPOCHS=20
BATCH_SIZE=8

MODEL_NAME=DeepSeek-R1-Distill-Qwen-7B
BASE_MODEL_PATH=deepseek-r1/${MODEL_NAME}
DATA_PATH=~/dataset/eagle-processed/OpenThoughts2-${MODEL_NAME}
CKPT_PATH=your-save-path


deepspeed eagle3_trainer.py \
    --deepspeed_config config/deepspeed_config.json \
    --base_model_path $BASE_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $CKPT_PATH \
    --project_name $PROJECT_NAME \
    --experiment_name $EXPERIMENT_NAME \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --precision bf16 \
    --max_len $MAX_LEN  \
    --freq_map_path freq_map/llama3/freq_32768.pt
