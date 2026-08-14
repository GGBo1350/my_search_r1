#!/usr/bin/env bash
# Sequential fixed-200 greedy evaluation for the continual Sample-K3 student:
#   Bridge checkpoints: global_step_25, global_step_50, global_step_75
#   Final checkpoint:   global_step_100 after the Comparison stage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
MODEL_PATH=${MODEL_PATH:-${ARTIFACT_ROOT}/models/Qwen--Qwen3-4B/snapshots/master}
BRIDGE_STEPS=${BRIDGE_STEPS:-"25 50 75"}
FINAL_STEP=${FINAL_STEP:-100}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_phase2_serial_k3_eval}

# The evaluator invokes the selected Python interpreter by absolute path, but
# FlashInfer launches the companion `ninja` executable through PATH for JIT
# kernels that are not covered by its prebuilt set.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=8
fi
if ! command -v ninja >/dev/null; then
    echo "ninja is missing from PATH; FlashInfer cannot compile required kernels." >&2
    exit 2
fi

if [[ -z "${SERIAL_RUN_ID:-}" ]]; then
    FINAL_CHECKPOINT_DIR=$(
        find "${ARTIFACT_ROOT}/checkpoints" \
            -maxdepth 1 -type d \
            -name 'qwen3_4b_opd_serial_k3_bridge_then_compare_s100_all7_lora_1gpu_*' \
            -print 2>/dev/null \
        | sort \
        | tail -n 1
    )
    if [[ -z "${FINAL_CHECKPOINT_DIR}" ]]; then
        echo "Cannot find a completed serial Sample-K3 checkpoint directory." >&2
        echo "Set SERIAL_RUN_ID explicitly if the experiment used a custom name." >&2
        exit 1
    fi
    SERIAL_RUN_ID=${FINAL_CHECKPOINT_DIR##*_1gpu_}
fi

BRIDGE_EXPERIMENT_NAME=${BRIDGE_EXPERIMENT_NAME:-qwen3_4b_opd_serial_k3_bridge_s75_all7_lora_1gpu_${SERIAL_RUN_ID}}
FINAL_EXPERIMENT_NAME=${FINAL_EXPERIMENT_NAME:-qwen3_4b_opd_serial_k3_bridge_then_compare_s100_all7_lora_1gpu_${SERIAL_RUN_ID}}
BRIDGE_CHECKPOINT_DIR=${BRIDGE_CHECKPOINT_DIR:-${ARTIFACT_ROOT}/checkpoints/${BRIDGE_EXPERIMENT_NAME}}
FINAL_CHECKPOINT_DIR=${FINAL_CHECKPOINT_DIR:-${ARTIFACT_ROOT}/checkpoints/${FINAL_EXPERIMENT_NAME}}
BRIDGE_TRAIN_LOG=${BRIDGE_TRAIN_LOG:-${ARTIFACT_ROOT}/train_logs/${BRIDGE_EXPERIMENT_NAME}.launch.log}
FINAL_TRAIN_LOG=${FINAL_TRAIN_LOG:-${ARTIFACT_ROOT}/train_logs/${FINAL_EXPERIMENT_NAME}.launch.log}

EVAL_RUN_ID=${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
EVAL_ROOT=${EVAL_ROOT:-${ARTIFACT_ROOT}/rollouts/serial_k3_ckpt_eval_${SERIAL_RUN_ID}_${EVAL_RUN_ID}}
EVAL_LOG_ROOT=${EVAL_LOG_ROOT:-${ARTIFACT_ROOT}/train_logs/serial_k3_ckpt_eval_${SERIAL_RUN_ID}_${EVAL_RUN_ID}}
REPORT_ROOT=${REPORT_ROOT:-${ARTIFACT_ROOT}/eval_reports/serial_k3_ckpt_eval_${SERIAL_RUN_ID}_${EVAL_RUN_ID}}

EVAL_ENTRY=recipe/eval/run_fixed200_after_training.sh
ANALYZER=recipe/phase1/analyze_full_lora_checkpoints.py

for required_file in "${EVAL_ENTRY}" "${ANALYZER}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required entrypoint is missing: ${required_file}" >&2
        exit 2
    fi
done
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Base model directory is missing: ${MODEL_PATH}" >&2
    exit 2
fi

for experiment_name in "${BRIDGE_EXPERIMENT_NAME}" "${FINAL_EXPERIMENT_NAME}"; do
    if pgrep -f "trainer.experiment_name=${experiment_name}" >/dev/null; then
        echo "Training is still running: ${experiment_name}" >&2
        exit 3
    fi
done

read -r -a bridge_step_list <<<"${BRIDGE_STEPS}"
all_steps=("${bridge_step_list[@]}" "${FINAL_STEP}")

actor_dir_for_step() {
    local step=$1
    if [[ "${step}" == "${FINAL_STEP}" ]]; then
        printf '%s\n' "${FINAL_CHECKPOINT_DIR}/global_step_${step}/actor"
    else
        printf '%s\n' "${BRIDGE_CHECKPOINT_DIR}/global_step_${step}/actor"
    fi
}

checkpoint_root_for_step() {
    local step=$1
    if [[ "${step}" == "${FINAL_STEP}" ]]; then
        printf '%s\n' "${FINAL_CHECKPOINT_DIR}"
    else
        printf '%s\n' "${BRIDGE_CHECKPOINT_DIR}"
    fi
}

experiment_name_for_step() {
    local step=$1
    if [[ "${step}" == "${FINAL_STEP}" ]]; then
        printf '%s\n' "${FINAL_EXPERIMENT_NAME}"
    else
        printf '%s\n' "${BRIDGE_EXPERIMENT_NAME}"
    fi
}

train_log_for_step() {
    local step=$1
    if [[ "${step}" == "${FINAL_STEP}" ]]; then
        printf '%s\n' "${FINAL_TRAIN_LOG}"
    else
        printf '%s\n' "${BRIDGE_TRAIN_LOG}"
    fi
}

for step in "${all_steps[@]}"; do
    actor_dir=$(actor_dir_for_step "${step}")
    if [[ ! -d "${actor_dir}" ]]; then
        echo "Checkpoint actor directory is missing: ${actor_dir}" >&2
        exit 4
    fi
    if [[ -z "$(find "${actor_dir}" -type f -size +0c -print -quit)" ]]; then
        echo "Checkpoint actor directory has no non-empty files: ${actor_dir}" >&2
        exit 4
    fi
done

mkdir -p "${EVAL_ROOT}" "${EVAL_LOG_ROOT}" "${REPORT_ROOT}"

# Match the established Phase 1/teacher fixed-200 protocol exactly.
export TRAIN_FILE=${TRAIN_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/train.parquet}
export TEST_FILE=${TEST_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/validation.parquet}
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}
export LORA_RANK=32
export LORA_ALPHA=64
export LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
export LORA_LAYERED_SUMMON=False
export ROLLOUT_LOAD_FORMAT=safetensors
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
export AGENT_TOOL_GPU_DEVICES=${AGENT_TOOL_GPU_DEVICES:-null}
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

analysis_inputs=()
for step in "${all_steps[@]}"; do
    checkpoint_root=$(checkpoint_root_for_step "${step}")
    experiment_name=$(experiment_name_for_step "${step}")
    train_log=$(train_log_for_step "${step}")
    eval_name="serial_k3_s${step}_fixed200_${EVAL_RUN_ID}"
    output_dir="${EVAL_ROOT}/s${step}"
    validation_log="${EVAL_LOG_ROOT}/s${step}.log"

    shopt -s nullglob
    existing_jsonl_files=("${output_dir}"/*.jsonl)
    shopt -u nullglob
    if (( ${#existing_jsonl_files[@]} > 0 )); then
        existing_row_count=$(awk 'END {print NR}' "${existing_jsonl_files[@]}")
        if [[ "${existing_row_count}" -eq 200 ]]; then
            echo "Reusing completed s${step} evaluation: ${output_dir}"
            analysis_inputs+=("s${step}=${output_dir}/*.jsonl")
            continue
        fi
        echo "Existing s${step} evaluation is incomplete: ${existing_row_count} rows." >&2
        echo "Use a new EVAL_RUN_ID; partial JSONL files are never overwritten." >&2
        exit 6
    fi

    if [[ -e "${validation_log}" ]]; then
        retry=1
        while [[ -e "${EVAL_LOG_ROOT}/s${step}.retry${retry}.log" ]]; do
            ((retry += 1))
        done
        validation_log="${EVAL_LOG_ROOT}/s${step}.retry${retry}.log"
    fi

    echo "==== Evaluating s${step}: ${checkpoint_root}/global_step_${step} ===="
    TRAIN_EXPERIMENT_NAME="${experiment_name}" \
    TRAIN_LOG="${train_log}" \
    CHECKPOINT_ROOT="${checkpoint_root}" \
    TARGET_STEP="${step}" \
    SKIP_PROGRESS_CHECK=True \
    VALIDATION_EXPERIMENT_NAME="${eval_name}" \
    VALIDATION_LOG="${validation_log}" \
    VALIDATION_OUTPUT_DIR="${output_dir}" \
        bash "${EVAL_ENTRY}"

    shopt -s nullglob
    jsonl_files=("${output_dir}"/*.jsonl)
    shopt -u nullglob
    if (( ${#jsonl_files[@]} == 0 )); then
        echo "Evaluation finished without a JSONL dump: ${output_dir}" >&2
        exit 7
    fi
    row_count=$(awk 'END {print NR}' "${jsonl_files[@]}")
    if [[ "${row_count}" -ne 200 ]]; then
        echo "Invalid evaluation row count for s${step}: ${row_count} != 200" >&2
        exit 7
    fi
    analysis_inputs+=("s${step}=${output_dir}/*.jsonl")
done

"${PYTHON_BIN}" "${ANALYZER}" \
    "${analysis_inputs[@]}" \
    --expected-count 200 \
    --json-output "${REPORT_ROOT}/summary.json" \
    --csv-output "${REPORT_ROOT}/summary.csv" \
    --text-output "${REPORT_ROOT}/summary.txt" \
    | tee "${REPORT_ROOT}/analysis.log"

echo "EVAL_DONE at $(date)"
echo "SERIAL_RUN_ID=${SERIAL_RUN_ID}"
echo "EVAL_ROOT=${EVAL_ROOT}"
echo "EVAL_LOG_ROOT=${EVAL_LOG_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
