#!/usr/bin/env bash
# Comparison specialist teacher: all seven Qwen3 LoRA projections, 25 steps.
# Checkpoint: global_step_25.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
FULL_LORA_TARGETS='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'

if [[ "${TEACHER_CONFIG_ONLY:-0}" != "1" ]] && ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "No NVIDIA GPU is attached. Enable the GPU on 931 before training." >&2
    exit 5
fi

mkdir -p "${ARTIFACT_ROOT}/train_logs" "${ARTIFACT_ROOT}/checkpoints" "${ARTIFACT_ROOT}/rollouts"
LOG_FILE="${ARTIFACT_ROOT}/train_logs/qwen3_4b_teacher_compare_full_lora_931_${RUN_ID}.launch.log"

cd "${PROJECT_ROOT}"
export PATH=/root/miniconda3/envs/verl/bin:${PATH}
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}

export MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen--Qwen3-4B/snapshots/master}
export TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3_hard_1600/train_compare_400.parquet}
export TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3_hard_1600/validation.parquet}
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}

export ROLLOUT_NAME=sglang
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export ROLLOUT_SKIP_TOKENIZER_INIT=False
export ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-flashinfer}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-2}

export LORA_RANK=32
export LORA_ALPHA=64
export LORA_TARGET_MODULES="${FULL_LORA_TARGETS}"
export TOTAL_TRAINING_STEPS=25
export SAVE_FREQ=25
export MAX_ACTOR_CKPT_TO_KEEP=1
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=-1
export RESUME_MODE=disable
export ANSWER_LLM_JUDGE=0

export ROLLOUT_DATA_ENABLED=True
export ROLLOUT_DATA_FREQ=${ROLLOUT_DATA_FREQ:-10}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${ARTIFACT_ROOT}/checkpoints/qwen3_4b_teacher_compare_full_lora_931_${RUN_ID}"}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${ARTIFACT_ROOT}/rollouts/qwen3_4b_teacher_compare_full_lora_931_${RUN_ID}"}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_teacher_full_lora}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_teacher_compare_full_lora_931_${RUN_ID}}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}

if [[ "${TEACHER_CONFIG_ONLY:-0}" != "1" ]]; then
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "ROLE=compare"
echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} SAVE_FREQ=${SAVE_FREQ} KEEP=${MAX_ACTOR_CKPT_TO_KEEP}"
echo "LORA_RANK=${LORA_RANK} LORA_ALPHA=${LORA_ALPHA}"
echo "LORA_TARGET_MODULES=${LORA_TARGET_MODULES}"
echo "ANSWER_LLM_JUDGE=${ANSWER_LLM_JUDGE}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

if [[ "${TEACHER_CONFIG_ONLY:-0}" == "1" ]]; then
    echo "Configuration-only check complete; training was not started."
    exit 0
fi

echo "TRAIN_START at $(date)"
bash recipe/train_lora/run_sglang_lora_100step.sh \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    "$@"
echo "TRAIN_DONE at $(date)"
