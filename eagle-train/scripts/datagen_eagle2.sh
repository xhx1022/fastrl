
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

MODEL_NAME=DeepSeek-R1-Distill-Qwen-7B
BASE_MODEL_PATH=deepseek-r1/${MODEL_NAME}
DATA_PATH=OpenThoughts2-1M
SAVE_DIR=~/dataset/eagle-processed/OpenThoughts2-${MODEL_NAME}


torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    eagle_datagen.py \
    model.base_model_path=$BASE_MODEL_PATH \
    data.data_path=$DATA_PATH \
    data.save_dir=$SAVE_DIR \
    data.max_length=2048 \
    data.sample_ratio=0.1 \
    data.mode=eagle2