#!/usr/bin/env bash
# Evaluate compare-teacher checkpoints greedily on the fixed 200-example set.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/my_search_r1}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:?must provide CHECKPOINT_ROOT}
TRAIN_LOG=${TRAIN_LOG:?must provide TRAIN_LOG}
TRAIN_EXPERIMENT_NAME=${TRAIN_EXPERIMENT_NAME:?must provide TRAIN_EXPERIMENT_NAME}
STEPS=${STEPS:-"20 25 30 35"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/rollouts}
LOG_ROOT=${LOG_ROOT:-/root/autodl-tmp/train_logs}

cd "${PROJECT_ROOT}"
export PATH="/root/miniconda3/envs/verl/bin:${PATH}"
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
export ROLLOUT_NAME=${ROLLOUT_NAME:-sglang}
export ROLLOUT_SKIP_TOKENIZER_INIT=${ROLLOUT_SKIP_TOKENIZER_INIT:-False}
export ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-flashinfer}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"[qkv_proj,o_proj,gate_up_proj,down_proj]"}
export MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen--Qwen3-4B/snapshots/master}
export TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3_hard_1600/train_compare_400.parquet}
export TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3_hard_1600/validation.parquet}
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

read -r -a step_list <<<"${STEPS}"
analysis_inputs=()
for step in "${step_list[@]}"; do
    output_dir="${OUTPUT_ROOT}/eval_teacher_compare_s${step}_greedy"
    TARGET_STEP="${step}" \
    VALIDATION_LOG="${LOG_ROOT}/eval_teacher_compare_s${step}_greedy.log" \
    VALIDATION_OUTPUT_DIR="${output_dir}" \
        bash recipe/eval/run_fixed200_after_training.sh
    analysis_inputs+=("s${step}=${output_dir}/*.jsonl")
done

python recipe/phase2/analyze_teacher_checkpoints.py "${analysis_inputs[@]}"
