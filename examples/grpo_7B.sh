export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0
export MKL_SERVICE_FORCE_INTEL=1
# export RAY_DEBUG_POST_MORTEM=1

CKPT_PATH=/root/projects/fastrl/checkpoint

PROJECT_NAME=FastRL
RANDOM_SUFFIX=$(date +%s%N | cut -b1-6)  
EXPERIMENT_NAME=Qwen2.5-7B-${RANDOM_SUFFIX}
MODEL_PATH=~/models/Qwen2.5-7B
DATA_PATH=~/datasets/Eurus-2-RL-Data
SPEC_MODEL_PATH=~/models/Qwen2.5-7B-Eagle-RL

train_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=4
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 32))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

ray stop --force
sleep 3

python3 -m verl.trainer.main_fastrl \
    speculative.eagle.spec_model_path=$SPEC_MODEL_PATH \
    speculative.enable=true \
    speculative.bs_threshold=32 \
    data.train_files=$DATA_PATH/filtered_train.parquet \
    data.val_files=$DATA_PATH/filtered_test.parquet \
    data.return_raw_chat=True \
    data.return_full_prompt=True \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1