#!/usr/bin/env bash
# Export and verify the full-LoRA Bridge/Comparison teacher adapters on 931.
# Existing complete adapters are verified and reused; nothing is overwritten.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
BASE_MODEL=${BASE_MODEL:-${ARTIFACT_ROOT}/models/Qwen--Qwen3-4B/snapshots/master}
TEACHER_PAIR_RUN_ID=${TEACHER_PAIR_RUN_ID:-20260813_113331}
FULL_LORA_TARGETS='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'

BRIDGE_TEACHER_ACTOR_CHECKPOINT=${BRIDGE_TEACHER_ACTOR_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/qwen3_4b_teacher_bridge_full_lora_931_${TEACHER_PAIR_RUN_ID}_bridge/global_step_75/actor}
COMPARE_TEACHER_ACTOR_CHECKPOINT=${COMPARE_TEACHER_ACTOR_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/qwen3_4b_teacher_compare_full_lora_931_${TEACHER_PAIR_RUN_ID}_compare/global_step_25/actor}
TEACHER_EXPORT_ROOT=${TEACHER_EXPORT_ROOT:-${ARTIFACT_ROOT}/models/search_r1_teacher_adapters/full_lora_${TEACHER_PAIR_RUN_ID}}
BRIDGE_TEACHER_EXPORT_ROOT=${BRIDGE_TEACHER_EXPORT_ROOT:-${TEACHER_EXPORT_ROOT}/bridge_s75}
COMPARE_TEACHER_EXPORT_ROOT=${COMPARE_TEACHER_EXPORT_ROOT:-${TEACHER_EXPORT_ROOT}/compare_s25}
BRIDGE_TEACHER_ADAPTER=${BRIDGE_TEACHER_ADAPTER:-${BRIDGE_TEACHER_EXPORT_ROOT}/lora_adapter}
COMPARE_TEACHER_ADAPTER=${COMPARE_TEACHER_ADAPTER:-${COMPARE_TEACHER_EXPORT_ROOT}/lora_adapter}

case "${EXTRACT_LOAD_DEVICE:-auto}" in
    auto)
        if nvidia-smi -L >/dev/null 2>&1; then
            EXTRACT_LOAD_DEVICE=cuda
        else
            echo "Automatic full-LoRA export requires an attached NVIDIA GPU." >&2
            echo "Attach the training GPUs, or explicitly set EXTRACT_LOAD_DEVICE=cpu if a slow CPU export is intentional." >&2
            exit 3
        fi
        ;;
    cpu|cuda|cuda:[0-9]*) EXTRACT_LOAD_DEVICE=${EXTRACT_LOAD_DEVICE} ;;
    *) echo "EXTRACT_LOAD_DEVICE must be auto, cpu, cuda, or cuda:N." >&2; exit 2 ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment is missing: ${PYTHON_BIN}" >&2
    exit 3
fi
if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
    echo "Base model is incomplete: ${BASE_MODEL}" >&2
    exit 3
fi

adapter_complete() {
    local adapter=$1
    [[ -s "${adapter}/adapter_config.json" && -s "${adapter}/adapter_model.safetensors" ]]
}

export_adapter() {
    local role=$1
    local checkpoint=$2
    local export_root=$3
    local adapter=$4

    if adapter_complete "${adapter}"; then
        echo "Reusing ${role} adapter: ${adapter}"
        return
    fi
    if [[ ! -d "${checkpoint}" ]]; then
        echo "${role} teacher actor checkpoint is missing: ${checkpoint}" >&2
        exit 4
    fi
    if [[ -d "${export_root}" ]] && find "${export_root}" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing to overwrite incomplete ${role} adapter export: ${export_root}" >&2
        exit 4
    fi
    if [[ "${EXTRACT_LOAD_DEVICE}" == cuda* ]] && ! nvidia-smi -L >/dev/null 2>&1; then
        echo "GPU extraction requested but no NVIDIA GPU is attached." >&2
        exit 4
    fi

    echo "Exporting ${role} teacher adapter from ${checkpoint} on ${EXTRACT_LOAD_DEVICE}."
    "${PYTHON_BIN}" recipe/phase2/extract_teacher_lora.py \
        --actor-checkpoint "${checkpoint}" \
        --base-model "${BASE_MODEL}" \
        --load-device "${EXTRACT_LOAD_DEVICE}" \
        --output "${export_root}"
}

export_adapter bridge "${BRIDGE_TEACHER_ACTOR_CHECKPOINT}" "${BRIDGE_TEACHER_EXPORT_ROOT}" "${BRIDGE_TEACHER_ADAPTER}"
export_adapter compare "${COMPARE_TEACHER_ACTOR_CHECKPOINT}" "${COMPARE_TEACHER_EXPORT_ROOT}" "${COMPARE_TEACHER_ADAPTER}"

"${PYTHON_BIN}" recipe/phase2/verify_teacher_adapters.py \
    --rank 32 \
    --target-modules "${FULL_LORA_TARGETS}" \
    "${BRIDGE_TEACHER_ADAPTER}" \
    "${COMPARE_TEACHER_ADAPTER}"

echo "TEACHER_ADAPTERS_READY"
echo "BRIDGE_TEACHER_ADAPTER=${BRIDGE_TEACHER_ADAPTER}"
echo "COMPARE_TEACHER_ADAPTER=${COMPARE_TEACHER_ADAPTER}"
