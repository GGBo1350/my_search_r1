#!/usr/bin/env bash
# Single-GPU greedy fixed-200 evaluation for the exported OPD student LoRA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STUDENT_EXPORT=${STUDENT_EXPORT:?provide an exported full seven-projection OPD student directory}
export STUDENT_EXPORT
export LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-${STUDENT_EXPORT}/lora_adapter}
export VALIDATION_EXPERIMENT_NAME=${VALIDATION_EXPERIMENT_NAME:-qwen3_4b_mopd_top32_p1024_s100_fixed200_greedy}
export VALIDATION_LOG=${VALIDATION_LOG:-/root/autodl-tmp/logs/${VALIDATION_EXPERIMENT_NAME}.log}
export VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR:-/root/autodl-tmp/rollouts/${VALIDATION_EXPERIMENT_NAME}}

if [[ ! -f "${LORA_ADAPTER_PATH}/adapter_config.json" || ! -f "${LORA_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
    echo "Exported LoRA adapter is incomplete: ${LORA_ADAPTER_PATH}" >&2
    exit 1
fi

export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
export MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen--Qwen3-4B/snapshots/master}
export TRAIN_FILE=${TRAIN_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/train_opd_routed.parquet}
export TEST_FILE=${TEST_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/validation.parquet}
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}

export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
export AGENT_TOOL_GPU_DEVICES=${AGENT_TOOL_GPU_DEVICES:-'[0]'}
export N_RESP_PER_PROMPT=1
export LORA_RANK=32
export LORA_ALPHA=64
export LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
"${PYTHON_BIN}" recipe/phase2/verify_teacher_adapters.py \
    --rank "${LORA_RANK}" \
    --target-modules "${LORA_TARGET_MODULES}" \
    "${LORA_ADAPTER_PATH}"
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=4096
export MAX_TOOL_RESPONSE_LENGTH=1024
export MAX_USER_TURNS=2
export MAX_ASSISTANT_TURNS=3
export MAX_PARALLEL_CALLS=2
export ROLLOUT_NAME=sglang
export ROLLOUT_SKIP_TOKENIZER_INIT=False
export ROLLOUT_ATTENTION_BACKEND=flashinfer
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.25}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
export ROLLOUT_MAX_MODEL_LEN=5121
export ROLLOUT_ENABLE_SLEEP_MODE=True
export ROLLOUT_FREE_CACHE_ENGINE=True
export VAL_BEFORE_TRAIN=True
export LOG_VAL_GENERATIONS=200
export ROLLOUT_DATA_ENABLED=False
export TEST_FREQ=-1
export SAVE_FREQ=-1
export RESUME_MODE=disable
export PROJECT_NAME=search_r1_hotpotqa_v3_mopd
export EXPERIMENT_NAME=${VALIDATION_EXPERIMENT_NAME}

/root/miniconda3/envs/verl/bin/ray stop --force >/dev/null 2>&1 || true
sleep 5
mkdir -p "$(dirname "${VALIDATION_LOG}")" "${VALIDATION_OUTPUT_DIR}"

exec bash recipe/train_lora/run_lora.sh \
    trainer.val_only=True \
    trainer.validation_data_dir="${VALIDATION_OUTPUT_DIR}" \
    actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}" \
    actor_rollout_ref.rollout.agent.tool_gpu_devices="${AGENT_TOOL_GPU_DEVICES}" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    >"${VALIDATION_LOG}" 2>&1
