#!/usr/bin/env bash
# Phase 2: train the bridge-specialist teacher.
# run_teacher_bridge_944.sh: Teacher-Bridge: 1200 bridge samples, 1 epoch = 75 steps, save every 20 (4 ckpts)
# One epoch of the Teacher-Bridge: 1200 bridge samples, 1 epoch = 75 steps, save every 20 (4 ckpts) split; general prompt kept in data (specialization comes from data only).
# Same hyperparams as Phase 1: bs16 x n8, mini 4, max_response 4096, LLM judge ON, chunked entropy.
RUN_ID=$(date +%Y%m%d_%H%M%S)
exec > /root/autodl-tmp/train_logs/qwen3_4b_teacher_bridge_944_${RUN_ID}.launch.log 2>&1
export PATH=/root/miniconda3/envs/verl/bin:$PATH
export FLASHINFER_ENABLE_AOT=1
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/my_search_r1}
cd "${PROJECT_ROOT}"
export ROLLOUT_NAME=sglang
export MODEL_PATH=/root/autodl-tmp/models/Qwen--Qwen3-4B/snapshots/master
export TRAIN_FILE=./data/hotpotqa_v3_hard_1600/train_bridge_1200.parquet
export TEST_FILE=./data/hotpotqa_v3_hard_1600/validation.parquet
export TOOL_CONFIG_PATH=recipe/core/tool_config_hybrid.yaml
export TRAIN_BATCH_SIZE=16
export N_RESP_PER_PROMPT=8
export PPO_MINI_BATCH_SIZE=4
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export ROLLOUT_GPU_MEM_UTIL=0.55
export ROLLOUT_SKIP_TOKENIZER_INIT=False
export ROLLOUT_ATTENTION_BACKEND=flashinfer
export LORA_RANK=32
export LORA_ALPHA=64
export LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
export AGENT_NUM_WORKERS=2
export REWARD_NUM_WORKERS=2
export TOTAL_TRAINING_STEPS=75
export SAVE_FREQ=20
export ROLLOUT_DATA_FREQ=10
export MAX_ACTOR_CKPT_TO_KEEP=4
export CHECKPOINT_DIR=/root/autodl-tmp/checkpoints/qwen3_4b_teacher_bridge_944_${RUN_ID}
export ROLLOUT_DATA_DIR=/root/autodl-tmp/rollouts/qwen3_4b_teacher_bridge_944_${RUN_ID}
export EXPERIMENT_NAME=qwen3_4b_teacher_bridge_944_${RUN_ID}
# LLM answer-equivalence judge ON
export ANSWER_LLM_JUDGE=${ANSWER_LLM_JUDGE:-1}
echo "launching run ${RUN_ID} (qwen3_4b_teacher_bridge_944) at $(date)"
bash recipe/train_lora/run_sglang_lora_100step.sh \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
echo "TRAIN_DONE at $(date)"
