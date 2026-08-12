#!/usr/bin/env bash
# 单卡 84 GB：Qwen3-4B + LoRA + GRPO + SGLang，正式训练 100 step。
# 本脚本不做训练前验证；训练完成后再单独运行固定 200 条验证。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
export MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen--Qwen3-4B/snapshots/master}
export TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3_2k/train.parquet}
export TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3_2k/validation.parquet}

export ROLLOUT_NAME=sglang
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export ACTOR_LR=${ACTOR_LR:-5e-6}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=-1
export SAVE_FREQ=${SAVE_FREQ:-100}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
export RESUME_MODE=${RESUME_MODE:-disable}

export ROLLOUT_DATA_ENABLED=True
export ROLLOUT_DATA_FREQ=${ROLLOUT_DATA_FREQ:-10}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-/root/autodl-tmp/rollouts/qwen3_4b_sglang_n8_bs8_100step}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/autodl-tmp/checkpoints/qwen3_4b_sglang_n8_bs8_100step}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_lora}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_sglang_lora_n8_bs8_100step}

# 84 GB 单卡先沿用已验证的保守上下文和并发上限；首步观察后再决定是否上调。
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-12288}

exec bash "${SCRIPT_DIR}/run_lora.sh" "$@"
