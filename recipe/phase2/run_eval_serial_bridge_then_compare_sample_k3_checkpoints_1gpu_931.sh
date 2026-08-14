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
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
PROJECT_NAME=${PROJECT_NAME:-search_r1_phase2_serial_k3_eval}

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

EVAL_RUN_ID=${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
EXPORT_ROOT=${EXPORT_ROOT:-${ARTIFACT_ROOT}/models/search_r1_student_adapters/serial_k3_${SERIAL_RUN_ID}}
EVAL_ROOT=${EVAL_ROOT:-${ARTIFACT_ROOT}/rollouts/serial_k3_ckpt_eval_${SERIAL_RUN_ID}_${EVAL_RUN_ID}}
EVAL_LOG_ROOT=${EVAL_LOG_ROOT:-${ARTIFACT_ROOT}/train_logs/serial_k3_ckpt_eval_${SERIAL_RUN_ID}_${EVAL_RUN_ID}}
REPORT_ROOT=${REPORT_ROOT:-${ARTIFACT_ROOT}/eval_reports/serial_k3_ckpt_eval_${SERIAL_RUN_ID}_${EVAL_RUN_ID}}

EXPORT_ENTRY=recipe/phase2/extract_teacher_lora.py
EVAL_ENTRY=recipe/phase2/run_eval_mopd_student_931.sh
ANALYZER=recipe/phase1/analyze_full_lora_checkpoints.py

for required_file in "${EXPORT_ENTRY}" "${EVAL_ENTRY}" "${ANALYZER}"; do
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

mkdir -p "${EXPORT_ROOT}" "${EVAL_ROOT}" "${EVAL_LOG_ROOT}" "${REPORT_ROOT}"
/root/miniconda3/envs/verl/bin/ray stop --force >/dev/null 2>&1 || true

for step in "${all_steps[@]}"; do
    actor_dir=$(actor_dir_for_step "${step}")
    export_dir="${EXPORT_ROOT}/s${step}"
    adapter_file="${export_dir}/lora_adapter/adapter_model.safetensors"

    if [[ -f "${adapter_file}" ]]; then
        echo "Reusing exported adapter for s${step}: ${export_dir}"
        continue
    fi
    if [[ -e "${export_dir}" ]]; then
        echo "Incomplete export directory already exists: ${export_dir}" >&2
        echo "Use a different EXPORT_ROOT or inspect the partial export manually." >&2
        exit 5
    fi

    echo "==== Exporting s${step}: ${actor_dir} ===="
    "${PYTHON_BIN}" "${EXPORT_ENTRY}" \
        --actor-checkpoint "${actor_dir}" \
        --base-model "${MODEL_PATH}" \
        --load-device cuda \
        --output "${export_dir}"
done

analysis_inputs=()
for step in "${all_steps[@]}"; do
    eval_name="serial_k3_s${step}_fixed200_${EVAL_RUN_ID}"
    output_dir="${EVAL_ROOT}/s${step}"
    validation_log="${EVAL_LOG_ROOT}/s${step}.log"

    if compgen -G "${output_dir}/*.jsonl" >/dev/null; then
        echo "Refusing to overwrite an existing evaluation dump: ${output_dir}" >&2
        exit 6
    fi

    echo "==== Evaluating s${step}: ${eval_name} ===="
    STUDENT_EXPORT="${EXPORT_ROOT}/s${step}" \
    VALIDATION_EXPERIMENT_NAME="${eval_name}" \
    VALIDATION_LOG="${validation_log}" \
    VALIDATION_OUTPUT_DIR="${output_dir}" \
    TRAINER_LOGGER="${TRAINER_LOGGER}" \
    PROJECT_NAME="${PROJECT_NAME}" \
    ANSWER_LLM_JUDGE=0 \
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
