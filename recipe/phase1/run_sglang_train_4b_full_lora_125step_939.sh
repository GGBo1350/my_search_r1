#!/usr/bin/env bash
# Phase 1: Qwen3-4B + full-module LoRA + GRPO + SGLang on 939.
# Save steps 25/50/75/100/125 and disable the external LLM answer judge.

set -euo pipefail

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/my_search_r1}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
MIN_FREE_GB=${MIN_FREE_GB:-55}

mkdir -p "${ARTIFACT_ROOT}/train_logs" "${ARTIFACT_ROOT}/checkpoints" "${ARTIFACT_ROOT}/rollouts"

AVAILABLE_KB=$(df -Pk "${ARTIFACT_ROOT}" | awk 'NR == 2 {print $4}')
REQUIRED_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( AVAILABLE_KB < REQUIRED_KB )); then
    AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
    echo "Insufficient checkpoint space: ${AVAILABLE_GB} GiB available, ${MIN_FREE_GB} GiB required." >&2
    echo "Expand ${ARTIFACT_ROOT} before launching the 125-step run." >&2
    exit 3
fi

if [[ "${PHASE1_CONFIG_ONLY:-0}" != "1" ]] && ! nvidia-smi -L >/dev/null 2>&1; then
    echo "No NVIDIA GPU is attached. Switch 939 out of no-card mode before training." >&2
    exit 5
fi

LOG_FILE="${ARTIFACT_ROOT}/train_logs/qwen3_4b_phase1_full_lora_125step_939_${RUN_ID}.launch.log"
exec > "${LOG_FILE}" 2>&1

export PATH=/root/miniconda3/envs/verl/bin:${PATH}
export FLASHINFER_ENABLE_AOT=1
cd "${PROJECT_ROOT}"

# Keep the proven Phase 1 rollout/training settings for a controlled comparison.
export ROLLOUT_NAME=sglang
export MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen--Qwen3-4B/snapshots/master}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export ROLLOUT_SKIP_TOKENIZER_INIT=False
export ROLLOUT_ATTENTION_BACKEND=flashinfer
export TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3_hard_1600/train.parquet}
export TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3_hard_1600/validation.parquet}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-2}

# Qwen3 uses separate projection names. The historical fused names qkv_proj and
# gate_up_proj matched nothing, leaving only o_proj/down_proj trainable.
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'

export TOTAL_TRAINING_STEPS=125
export SAVE_FREQ=25
export MAX_ACTOR_CKPT_TO_KEEP=5
export RESUME_MODE=disable
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=-1
export ANSWER_LLM_JUDGE=0

export ROLLOUT_DATA_ENABLED=True
export ROLLOUT_DATA_FREQ=${ROLLOUT_DATA_FREQ:-10}
export CHECKPOINT_DIR="${ARTIFACT_ROOT}/checkpoints/qwen3_4b_phase1_full_lora_125step_939_${RUN_ID}"
export ROLLOUT_DATA_DIR="${ARTIFACT_ROOT}/rollouts/qwen3_4b_phase1_full_lora_125step_939_${RUN_ID}"
export PROJECT_NAME=${PROJECT_NAME:-search_r1_phase1_full_lora}
export EXPERIMENT_NAME="qwen3_4b_phase1_full_lora_125step_939_${RUN_ID}"
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}

# Support both the reorganized main tree and the older 903/907 clone layout.
if [[ -f recipe/train_lora/run_sglang_lora_100step.sh ]]; then
    TRAIN_ENTRY=recipe/train_lora/run_sglang_lora_100step.sh
    export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}
elif [[ -f recipe/v3/run_sglang_lora_100step.sh ]]; then
    TRAIN_ENTRY=recipe/v3/run_sglang_lora_100step.sh
    export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/v3/tool_config_hybrid.yaml}
else
    echo "Cannot find the Phase 1 SGLang LoRA training entrypoint." >&2
    exit 4
fi

echo "RUN_ID=${RUN_ID}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "TRAIN_ENTRY=${TRAIN_ENTRY}"
echo "LORA_TARGET_MODULES=${LORA_TARGET_MODULES}"
echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} SAVE_FREQ=${SAVE_FREQ} KEEP=${MAX_ACTOR_CKPT_TO_KEEP}"
echo "ANSWER_LLM_JUDGE=${ANSWER_LLM_JUDGE} VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN} TEST_FREQ=${TEST_FREQ}"
echo "TRAINER_LOGGER=${TRAINER_LOGGER}"

if [[ "${PHASE1_CONFIG_ONLY:-0}" == "1" ]]; then
    echo "Configuration-only check complete; training was not started."
    exit 0
fi

bash "${TRAIN_ENTRY}" \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True

echo "TRAIN_DONE at $(date)"
