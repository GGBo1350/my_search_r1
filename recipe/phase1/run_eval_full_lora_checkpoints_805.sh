#!/usr/bin/env bash
# Sequential fixed-200 greedy evaluation for all checkpoints from the current
# Phase 1 full-module LoRA run.  Each checkpoint gets its own SwanLab run.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/my_search_r1}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
TRAIN_RUN_ID=${TRAIN_RUN_ID:-20260811_104813}
TRAIN_EXPERIMENT_NAME=${TRAIN_EXPERIMENT_NAME:-qwen3_4b_phase1_full_lora_125step_939_${TRAIN_RUN_ID}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${ARTIFACT_ROOT}/checkpoints/${TRAIN_EXPERIMENT_NAME}}
TRAIN_LOG=${TRAIN_LOG:-${ARTIFACT_ROOT}/train_logs/${TRAIN_EXPERIMENT_NAME}.launch.log}
STEPS=${STEPS:-"25 50 75 100 125"}
POLL_SECONDS=${POLL_SECONDS:-30}
WAIT_FOR_TRAIN=${WAIT_FOR_TRAIN:-True}
EVAL_RUN_ID=${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
EVAL_ROOT=${EVAL_ROOT:-${ARTIFACT_ROOT}/rollouts/phase1_full_lora_ckpt_eval_${EVAL_RUN_ID}}
EVAL_LOG_ROOT=${EVAL_LOG_ROOT:-${ARTIFACT_ROOT}/train_logs/phase1_full_lora_ckpt_eval_${EVAL_RUN_ID}}
REPORT_ROOT=${REPORT_ROOT:-${ARTIFACT_ROOT}/eval_reports/phase1_full_lora_ckpt_eval_${EVAL_RUN_ID}}

case "${WAIT_FOR_TRAIN,,}" in
    true|1|yes)
        echo "Waiting for training process: ${TRAIN_EXPERIMENT_NAME}"
        while pgrep -f "trainer.experiment_name=${TRAIN_EXPERIMENT_NAME}" >/dev/null; do
            sleep "${POLL_SECONDS}"
        done
        # The launcher writes TRAIN_DONE immediately after main_ppo exits.  Give
        # that parent shell a short grace period so we do not fail on the race.
        for _ in {1..30}; do
            grep -aq "TRAIN_DONE" "${TRAIN_LOG}" && break
            sleep 2
        done
        if ! grep -aq "TRAIN_DONE" "${TRAIN_LOG}"; then
            echo "Training exited without TRAIN_DONE; refusing to start evaluation: ${TRAIN_LOG}" >&2
            exit 1
        fi
        ;;
    false|0|no) : ;;
    *) echo "WAIT_FOR_TRAIN must be True or False, got: ${WAIT_FOR_TRAIN}" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p "${EVAL_ROOT}" "${EVAL_LOG_ROOT}" "${REPORT_ROOT}"

# Support both the reorganized main tree and the older 805 clone layout.
if [[ -f recipe/eval/run_fixed200_after_training.sh ]]; then
    FIXED200_ENTRY=recipe/eval/run_fixed200_after_training.sh
    DEFAULT_TOOL_CONFIG=recipe/core/tool_config_hybrid.yaml
elif [[ -f recipe/v3/run_fixed200_after_training.sh ]]; then
    FIXED200_ENTRY=recipe/v3/run_fixed200_after_training.sh
    DEFAULT_TOOL_CONFIG=recipe/v3/tool_config_hybrid.yaml
else
    echo "Cannot find the fixed-200 evaluation entrypoint." >&2
    exit 4
fi

export PATH="/root/miniconda3/envs/verl/bin:${PATH}"
export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
export MODEL_PATH=${MODEL_PATH:-${ARTIFACT_ROOT}/models/Qwen--Qwen3-4B/snapshots/master}
export TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3_hard_1600/train.parquet}
export TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3_hard_1600/validation.parquet}
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-${DEFAULT_TOOL_CONFIG}}

# Match the training architecture exactly.
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'}
export LORA_LAYERED_SUMMON=False
export ROLLOUT_LOAD_FORMAT=safetensors

# Deterministic fixed-200 evaluation settings.
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
# Each search worker loads its own retrieval model on the GPU.  Eight workers
# exhausted the 96 GB card alongside FSDP and SGLang, causing tool-call OOMs.
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-4}
export N_RESP_PER_PROMPT=1
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-3072}
export MAX_USER_TURNS=${MAX_USER_TURNS:-3}
export MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-4}
export MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-2}
export ROLLOUT_NAME=sglang
export ROLLOUT_SKIP_TOKENIZER_INIT=False
export ROLLOUT_ATTENTION_BACKEND=flashinfer
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-12288}
export ROLLOUT_ENABLE_SLEEP_MODE=False
export ROLLOUT_FREE_CACHE_ENGINE=False
export ACTOR_MODEL_DTYPE=bfloat16
export REF_MODEL_DTYPE=bfloat16
export ANSWER_LLM_JUDGE=0
export LOG_VAL_GENERATIONS=200
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_phase1_full_lora_eval}

read -r -a step_list <<<"${STEPS}"
analysis_inputs=()
for step in "${step_list[@]}"; do
    step_dir="${CHECKPOINT_ROOT}/global_step_${step}"
    if [[ ! -d "${step_dir}/actor" ]]; then
        echo "Checkpoint is incomplete or missing: ${step_dir}/actor" >&2
        exit 1
    fi
    if ! find "${step_dir}/actor" -type f -size +0c -print -quit | grep -q .; then
        echo "Checkpoint actor directory contains no non-empty files: ${step_dir}/actor" >&2
        exit 1
    fi

    eval_name="phase1_full_lora_s${step}_fixed200_greedy_${EVAL_RUN_ID}"
    output_dir="${EVAL_ROOT}/s${step}"
    validation_log="${EVAL_LOG_ROOT}/s${step}.log"
    if find "${output_dir}" -maxdepth 1 -name '*.jsonl' -type f -size +0c -print -quit 2>/dev/null | grep -q .; then
        echo "Refusing to overwrite an existing evaluation dump: ${output_dir}" >&2
        echo "Use a new EVAL_RUN_ID or remove the old output explicitly." >&2
        exit 1
    fi

    echo "==== Evaluating global_step_${step} (${eval_name}) ===="
    TRAIN_EXPERIMENT_NAME="${TRAIN_EXPERIMENT_NAME}" \
    TRAIN_LOG="${TRAIN_LOG}" \
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
    TARGET_STEP="${step}" \
    SKIP_PROGRESS_CHECK=True \
    VALIDATION_EXPERIMENT_NAME="${eval_name}" \
    VALIDATION_LOG="${validation_log}" \
    VALIDATION_OUTPUT_DIR="${output_dir}" \
        bash "${FIXED200_ENTRY}"

    if ! find "${output_dir}" -maxdepth 1 -name '*.jsonl' -type f -size +0c -print -quit | grep -q .; then
        echo "Evaluation finished without a non-empty JSONL dump: ${output_dir}" >&2
        exit 1
    fi
    analysis_inputs+=("s${step}=${output_dir}/*.jsonl")
done

"${PYTHON_BIN}" recipe/phase1/analyze_full_lora_checkpoints.py \
    "${analysis_inputs[@]}" \
    --expected-count 200 \
    --json-output "${REPORT_ROOT}/summary.json" \
    --csv-output "${REPORT_ROOT}/summary.csv" \
    --text-output "${REPORT_ROOT}/summary.txt" \
    | tee "${REPORT_ROOT}/analysis.log"

echo "EVAL_DONE at $(date)"
echo "EVAL_ROOT=${EVAL_ROOT}"
echo "EVAL_LOG_ROOT=${EVAL_LOG_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
