#!/usr/bin/env bash
# Phase 1 ablation: strict Exact Answer-only reward and no strategy prompt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

mkdir -p "${ARTIFACT_ROOT}/train_logs" "${ARTIFACT_ROOT}/checkpoints" "${ARTIFACT_ROOT}/rollouts"
LOG_FILE="${ARTIFACT_ROOT}/train_logs/qwen3_4b_exact_answer_only_50step_${RUN_ID}.launch.log"
exec >"${LOG_FILE}" 2>&1
cd "${PROJECT_ROOT}"

export MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen--Qwen3-4B/snapshots/master}
export TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3_no_strategy/train.parquet}
export TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3_no_strategy/validation.parquet}
export REWARD_PATH=recipe/core/my_reward_exact_only.py
export REWARD_NAME=compute_score
export ANSWER_LLM_JUDGE=0
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}

export ROLLOUT_NAME=sglang
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export ROLLOUT_SKIP_TOKENIZER_INIT=False
export ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-flashinfer}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-2}

# The public reproduction uses all seven Qwen3 projection families.
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'

export TOTAL_TRAINING_STEPS=50
export SAVE_FREQ=50
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=-1
export RESUME_MODE=disable
export ROLLOUT_DATA_ENABLED=True
export ROLLOUT_DATA_FREQ=${ROLLOUT_DATA_FREQ:-10}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${ARTIFACT_ROOT}/checkpoints/qwen3_4b_exact_answer_only_50step_${RUN_ID}"}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${ARTIFACT_ROOT}/rollouts/qwen3_4b_exact_answer_only_50step_${RUN_ID}"}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_exact_answer_only}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_exact_answer_only_50step_${RUN_ID}}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}

echo "RUN_ID=${RUN_ID}"
echo "REWARD_PATH=${REWARD_PATH} ANSWER_LLM_JUDGE=${ANSWER_LLM_JUDGE}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "LORA_TARGET_MODULES=${LORA_TARGET_MODULES}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

bash recipe/train_lora/run_sglang_lora_100step.sh \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    "$@"

echo "TRAIN_DONE at $(date)"

