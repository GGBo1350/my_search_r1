#!/usr/bin/env bash
# Sequential fixed-200 greedy evaluation for the full-LoRA Bridge/Comparison
# teacher pair trained by run_teacher_pair_full_lora_931.sh.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/my_search_r1}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
PAIR_RUN_ID=${PAIR_RUN_ID:?set PAIR_RUN_ID to the teacher-pair run id, for example 20260813_113331}
BRIDGE_STEPS=${BRIDGE_STEPS:-"25 50 75"}
COMPARE_STEPS=${COMPARE_STEPS:-"25"}
EVAL_RUN_ID=${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}

BRIDGE_RUN_ID=${BRIDGE_RUN_ID:-${PAIR_RUN_ID}_bridge}
COMPARE_RUN_ID=${COMPARE_RUN_ID:-${PAIR_RUN_ID}_compare}
BRIDGE_EXPERIMENT_NAME=${BRIDGE_EXPERIMENT_NAME:-qwen3_4b_teacher_bridge_full_lora_931_${BRIDGE_RUN_ID}}
COMPARE_EXPERIMENT_NAME=${COMPARE_EXPERIMENT_NAME:-qwen3_4b_teacher_compare_full_lora_931_${COMPARE_RUN_ID}}
BRIDGE_CHECKPOINT_ROOT=${BRIDGE_CHECKPOINT_ROOT:-${ARTIFACT_ROOT}/checkpoints/${BRIDGE_EXPERIMENT_NAME}}
COMPARE_CHECKPOINT_ROOT=${COMPARE_CHECKPOINT_ROOT:-${ARTIFACT_ROOT}/checkpoints/${COMPARE_EXPERIMENT_NAME}}
BRIDGE_TRAIN_LOG=${BRIDGE_TRAIN_LOG:-${ARTIFACT_ROOT}/train_logs/${BRIDGE_EXPERIMENT_NAME}.launch.log}
COMPARE_TRAIN_LOG=${COMPARE_TRAIN_LOG:-${ARTIFACT_ROOT}/train_logs/${COMPARE_EXPERIMENT_NAME}.launch.log}

EVAL_ROOT=${EVAL_ROOT:-${ARTIFACT_ROOT}/rollouts/teacher_pair_full_lora_931_eval_${PAIR_RUN_ID}_${EVAL_RUN_ID}}
EVAL_LOG_ROOT=${EVAL_LOG_ROOT:-${ARTIFACT_ROOT}/train_logs/teacher_pair_full_lora_931_eval_${PAIR_RUN_ID}_${EVAL_RUN_ID}}
REPORT_ROOT=${REPORT_ROOT:-${ARTIFACT_ROOT}/eval_reports/teacher_pair_full_lora_931_eval_${PAIR_RUN_ID}_${EVAL_RUN_ID}}

cd "${PROJECT_ROOT}"

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

if [[ -f recipe/phase1/analyze_full_lora_checkpoints.py ]]; then
    ANALYZER=recipe/phase1/analyze_full_lora_checkpoints.py
else
    echo "Cannot find recipe/phase1/analyze_full_lora_checkpoints.py." >&2
    exit 4
fi

for experiment_name in "${BRIDGE_EXPERIMENT_NAME}" "${COMPARE_EXPERIMENT_NAME}"; do
    if pgrep -f "trainer.experiment_name=${experiment_name}" >/dev/null; then
        echo "Training is still running: ${experiment_name}" >&2
        echo "Run this evaluation after the Comparison teacher finishes." >&2
        exit 3
    fi
done

for train_log in "${BRIDGE_TRAIN_LOG}" "${COMPARE_TRAIN_LOG}"; do
    if [[ ! -s "${train_log}" ]] || ! grep -aq "TRAIN_DONE" "${train_log}"; then
        echo "Training completion marker is missing: ${train_log}" >&2
        echo "Refusing to evaluate a possibly incomplete teacher run." >&2
        exit 3
    fi
done

mkdir -p "${EVAL_ROOT}" "${EVAL_LOG_ROOT}" "${REPORT_ROOT}"

export PATH="/root/miniconda3/envs/verl/bin:${PATH}"
export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
export MODEL_PATH=${MODEL_PATH:-${ARTIFACT_ROOT}/models/Qwen--Qwen3-4B/snapshots/master}
export TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3_hard_1600/train.parquet}
export TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3_hard_1600/validation.parquet}
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-${DEFAULT_TOOL_CONFIG}}

export LORA_RANK=32
export LORA_ALPHA=64
export LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
export LORA_LAYERED_SUMMON=False
export ROLLOUT_LOAD_FORMAT=safetensors

export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
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
export ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-flashinfer}
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
export PROJECT_NAME=${PROJECT_NAME:-search_r1_teacher_full_lora_eval}

analysis_inputs=()

evaluate_checkpoint() {
    local role=$1
    local step=$2
    local experiment_name=$3
    local checkpoint_root=$4
    local train_log=$5
    local step_dir="${checkpoint_root}/global_step_${step}"
    local label="${role}_s${step}"
    local eval_name="teacher_${label}_full_lora_fixed200_${PAIR_RUN_ID}_${EVAL_RUN_ID}"
    local output_dir="${EVAL_ROOT}/${label}"
    local validation_log="${EVAL_LOG_ROOT}/${label}.log"

    if [[ ! -d "${step_dir}/actor" ]]; then
        echo "Checkpoint actor directory is missing: ${step_dir}/actor" >&2
        exit 1
    fi
    if ! find "${step_dir}/actor" -type f -size +0c -print -quit | grep -q .; then
        echo "Checkpoint actor directory has no non-empty files: ${step_dir}/actor" >&2
        exit 1
    fi
    if find "${output_dir}" -maxdepth 1 -name '*.jsonl' -type f -size +0c -print -quit 2>/dev/null | grep -q .; then
        echo "Refusing to overwrite an existing evaluation dump: ${output_dir}" >&2
        echo "Use a new EVAL_RUN_ID to create a new evaluation." >&2
        exit 1
    fi

    echo "==== Evaluating ${label}: ${step_dir} ===="
    TRAIN_EXPERIMENT_NAME="${experiment_name}" \
    TRAIN_LOG="${train_log}" \
    CHECKPOINT_ROOT="${checkpoint_root}" \
    TARGET_STEP="${step}" \
    SKIP_PROGRESS_CHECK=True \
    VALIDATION_EXPERIMENT_NAME="${eval_name}" \
    VALIDATION_LOG="${validation_log}" \
    VALIDATION_OUTPUT_DIR="${output_dir}" \
        bash "${FIXED200_ENTRY}"

    if ! find "${output_dir}" -maxdepth 1 -name '*.jsonl' -type f -size +0c -print -quit 2>/dev/null | grep -q .; then
        echo "Evaluation finished without a non-empty JSONL dump: ${output_dir}" >&2
        exit 1
    fi
    analysis_inputs+=("${label}=${output_dir}/*.jsonl")
}

read -r -a bridge_step_list <<<"${BRIDGE_STEPS}"
for step in "${bridge_step_list[@]}"; do
    evaluate_checkpoint bridge "${step}" "${BRIDGE_EXPERIMENT_NAME}" "${BRIDGE_CHECKPOINT_ROOT}" "${BRIDGE_TRAIN_LOG}"
done

read -r -a compare_step_list <<<"${COMPARE_STEPS}"
for step in "${compare_step_list[@]}"; do
    evaluate_checkpoint compare "${step}" "${COMPARE_EXPERIMENT_NAME}" "${COMPARE_CHECKPOINT_ROOT}" "${COMPARE_TRAIN_LOG}"
done

"${PYTHON_BIN}" "${ANALYZER}" \
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
