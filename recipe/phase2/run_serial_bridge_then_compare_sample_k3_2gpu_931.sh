#!/usr/bin/env bash
# Continual two-stage Sample-K3 OPD on one student:
#   Qwen3-4B Base --Bridge teacher/1200 rows--> s75
#                 --Compare teacher/400 rows--> s100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
BASE_MODEL=${BASE_MODEL:-${ARTIFACT_ROOT}/models/Qwen--Qwen3-4B/snapshots/master}
TEACHER_PAIR_RUN_ID=${TEACHER_PAIR_RUN_ID:-20260813_113331}
TEACHER_EXPORT_ROOT=${TEACHER_EXPORT_ROOT:-${ARTIFACT_ROOT}/models/search_r1_teacher_adapters/full_lora_${TEACHER_PAIR_RUN_ID}}
BRIDGE_TEACHER_ADAPTER=${BRIDGE_TEACHER_ADAPTER:-${TEACHER_EXPORT_ROOT}/bridge_s75/lora_adapter}
COMPARE_TEACHER_ADAPTER=${COMPARE_TEACHER_ADAPTER:-${TEACHER_EXPORT_ROOT}/compare_s25/lora_adapter}

BRIDGE_TRAIN_FILE=${BRIDGE_TRAIN_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/train_bridge_1200.parquet}
COMPARE_TRAIN_FILE=${COMPARE_TRAIN_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/train_compare_400.parquet}
TEST_FILE=${TEST_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/validation.parquet}

BRIDGE_EXPERIMENT_NAME=${BRIDGE_EXPERIMENT_NAME:-qwen3_4b_opd_serial_k3_bridge_s75_all7_lora_2gpu_${RUN_ID}}
COMPARE_EXPERIMENT_NAME=${COMPARE_EXPERIMENT_NAME:-qwen3_4b_opd_serial_k3_bridge_then_compare_s100_all7_lora_2gpu_${RUN_ID}}
BRIDGE_CHECKPOINT_DIR=${BRIDGE_CHECKPOINT_DIR:-${ARTIFACT_ROOT}/checkpoints/${BRIDGE_EXPERIMENT_NAME}}
COMPARE_CHECKPOINT_DIR=${COMPARE_CHECKPOINT_DIR:-${ARTIFACT_ROOT}/checkpoints/${COMPARE_EXPERIMENT_NAME}}
BRIDGE_ROLLOUT_DIR=${BRIDGE_ROLLOUT_DIR:-${ARTIFACT_ROOT}/rollouts/${BRIDGE_EXPERIMENT_NAME}}
COMPARE_ROLLOUT_DIR=${COMPARE_ROLLOUT_DIR:-${ARTIFACT_ROOT}/rollouts/${COMPARE_EXPERIMENT_NAME}}
BRIDGE_LOG=${BRIDGE_LOG:-${ARTIFACT_ROOT}/train_logs/${BRIDGE_EXPERIMENT_NAME}.launch.log}
COMPARE_LOG=${COMPARE_LOG:-${ARTIFACT_ROOT}/train_logs/${COMPARE_EXPERIMENT_NAME}.launch.log}
MASTER_LOG=${MASTER_LOG:-${ARTIFACT_ROOT}/train_logs/opd_serial_bridge_then_compare_k3_${RUN_ID}.master.log}
HANDOFF_STEP=${HANDOFF_STEP:-${ARTIFACT_ROOT}/opd_handoffs/serial_bridge_then_compare_k3_${RUN_ID}/global_step_75}

available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if (( available_gpus < 2 )); then
    echo "This serial Sample-K3 experiment requires two visible GPUs; found ${available_gpus}." >&2
    exit 3
fi
if [[ "${TRAIN_BATCH_SIZE:-16}" != "16" ]]; then
    echo "This 75+25 step curriculum requires TRAIN_BATCH_SIZE=16." >&2
    exit 2
fi
for f in "${BRIDGE_TRAIN_FILE}" "${COMPARE_TRAIN_FILE}" "${TEST_FILE}"; do
    if [[ ! -f "${f}" ]]; then
        echo "Required dataset is missing: ${f}" >&2
        exit 3
    fi
done
for d in \
    "${BRIDGE_CHECKPOINT_DIR}" "${COMPARE_CHECKPOINT_DIR}" \
    "${BRIDGE_ROLLOUT_DIR}" "${COMPARE_ROLLOUT_DIR}" \
    "${BRIDGE_LOG}" "${COMPARE_LOG}" "${MASTER_LOG}" \
    "$(dirname "${HANDOFF_STEP}")"; do
    if [[ -e "${d}" ]]; then
        echo "Refusing to overwrite an existing serial-run path: ${d}" >&2
        exit 4
    fi
done

BRIDGE_ROWS=$("${PYTHON_BIN}" -c 'import pyarrow.parquet as pq, sys; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "${BRIDGE_TRAIN_FILE}")
COMPARE_ROWS=$("${PYTHON_BIN}" -c 'import pyarrow.parquet as pq, sys; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "${COMPARE_TRAIN_FILE}")
if [[ "${BRIDGE_ROWS}" != "1200" || "${COMPARE_ROWS}" != "400" ]]; then
    echo "Expected 1200 Bridge rows and 400 Comparison rows; got ${BRIDGE_ROWS} and ${COMPARE_ROWS}." >&2
    exit 3
fi

mkdir -p "$(dirname "${MASTER_LOG}")"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "RUN_ID=${RUN_ID}"
echo "CURRICULUM=bridge_s75_then_compare_s100"
echo "DISTILLATION_LOSS_MODE=k3"

ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
BASE_MODEL="${BASE_MODEL}" \
TEACHER_PAIR_RUN_ID="${TEACHER_PAIR_RUN_ID}" \
TEACHER_EXPORT_ROOT="${TEACHER_EXPORT_ROOT}" \
BRIDGE_TEACHER_ADAPTER="${BRIDGE_TEACHER_ADAPTER}" \
COMPARE_TEACHER_ADAPTER="${COMPARE_TEACHER_ADAPTER}" \
    bash "${SCRIPT_DIR}/prepare_full_lora_teacher_pair_931.sh"

echo "Starting Bridge Sample-K3 stage: 1200 rows, expected stop at global_step_75."
STAGE_NAME=bridge \
TEACHER_ADAPTER="${BRIDGE_TEACHER_ADAPTER}" \
TRAIN_FILE="${BRIDGE_TRAIN_FILE}" \
TEST_FILE="${TEST_FILE}" \
EXPERIMENT_NAME="${BRIDGE_EXPERIMENT_NAME}" \
CHECKPOINT_DIR="${BRIDGE_CHECKPOINT_DIR}" \
ROLLOUT_DATA_DIR="${BRIDGE_ROLLOUT_DIR}" \
LOG_FILE="${BRIDGE_LOG}" \
ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
BASE_MODEL="${BASE_MODEL}" \
STUDENT_MODEL="${BASE_MODEL}" \
TRAIN_BATCH_SIZE=16 TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=100 \
SAVE_FREQ=25 MAX_ACTOR_CKPT_TO_KEEP=3 \
    bash "${SCRIPT_DIR}/run_single_teacher_sample_k3_2gpu_931.sh" "$@"

BRIDGE_STEP="${BRIDGE_CHECKPOINT_DIR}/global_step_75"
for step in 25 50 75; do
    if [[ ! -d "${BRIDGE_CHECKPOINT_DIR}/global_step_${step}/actor" ]]; then
        echo "Bridge stage did not retain global_step_${step}." >&2
        exit 5
    fi
done

# Resume model/optimizer/LR-scheduler/RNG state only. The deliberately absent
# data.pt prevents the Bridge dataloader cursor from entering Comparison.
# The symlink avoids duplicating the actor checkpoint.
mkdir -p "${HANDOFF_STEP}"
ln -s "${BRIDGE_STEP}/actor" "${HANDOFF_STEP}/actor"
if [[ -e "${HANDOFF_STEP}/data.pt" ]]; then
    echo "Unexpected dataloader state in serial handoff: ${HANDOFF_STEP}/data.pt" >&2
    exit 5
fi

echo "Starting Comparison Sample-K3 stage: resume student at s75 and train through global_step_100."
# ray_trainer derives current_epoch as global_step / current dataloader length.
# For the 25-step Comparison loader, s75 maps to epoch 3; total_epochs=4 runs
# exactly one Comparison epoch instead of skipping the stage.
STAGE_NAME=compare \
TEACHER_ADAPTER="${COMPARE_TEACHER_ADAPTER}" \
TRAIN_FILE="${COMPARE_TRAIN_FILE}" \
TEST_FILE="${TEST_FILE}" \
EXPERIMENT_NAME="${COMPARE_EXPERIMENT_NAME}" \
CHECKPOINT_DIR="${COMPARE_CHECKPOINT_DIR}" \
ROLLOUT_DATA_DIR="${COMPARE_ROLLOUT_DIR}" \
LOG_FILE="${COMPARE_LOG}" \
RESUME_FROM_PATH="${HANDOFF_STEP}" \
ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
BASE_MODEL="${BASE_MODEL}" \
STUDENT_MODEL="${BASE_MODEL}" \
TRAIN_BATCH_SIZE=16 TOTAL_EPOCHS=4 TOTAL_TRAINING_STEPS=100 \
SAVE_FREQ=100 MAX_ACTOR_CKPT_TO_KEEP=1 \
    bash "${SCRIPT_DIR}/run_single_teacher_sample_k3_2gpu_931.sh" "$@"

if [[ ! -d "${COMPARE_CHECKPOINT_DIR}/global_step_100/actor" ]]; then
    echo "Comparison stage did not produce global_step_100." >&2
    exit 5
fi

echo "SERIAL_SAMPLE_K3_DONE at $(date)"
echo "BRIDGE_CHECKPOINTS=${BRIDGE_CHECKPOINT_DIR}/global_step_{25,50,75}"
echo "FINAL_CHECKPOINT=${COMPARE_CHECKPOINT_DIR}/global_step_100"
echo "MASTER_LOG=${MASTER_LOG}"
