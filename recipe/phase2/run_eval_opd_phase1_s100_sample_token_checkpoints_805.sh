#!/usr/bin/env bash
# Fixed-200 greedy evaluation for the four OPD sample-token checkpoints on 805.
# Each checkpoint is evaluated in a separate SwanLab run, then strict answer,
# retrieval and bridge/compare strategy metrics are summarized together.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
TRAIN_RUN_ID=${TRAIN_RUN_ID:-20260812_124332}
TRAIN_EXPERIMENT_NAME=${TRAIN_EXPERIMENT_NAME:-qwen3_4b_opd_phase1_s100_sample_token_k3_all7_lora_r32_1gpu_${TRAIN_RUN_ID}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${ARTIFACT_ROOT}/checkpoints/${TRAIN_EXPERIMENT_NAME}}
TRAIN_LOG=${TRAIN_LOG:-${ARTIFACT_ROOT}/train_logs/${TRAIN_EXPERIMENT_NAME}.launch.log}
STEPS=${STEPS:-"25 50 75 100"}
EVAL_RUN_ID=${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
EVAL_ROOT=${EVAL_ROOT:-${ARTIFACT_ROOT}/rollouts/opd_sample_token_ckpt_eval_${EVAL_RUN_ID}}
EVAL_LOG_ROOT=${EVAL_LOG_ROOT:-${ARTIFACT_ROOT}/train_logs/opd_sample_token_ckpt_eval_${EVAL_RUN_ID}}
REPORT_ROOT=${REPORT_ROOT:-${ARTIFACT_ROOT}/eval_reports/opd_sample_token_ckpt_eval_${EVAL_RUN_ID}}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}

FULL_LORA_TARGETS='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-${FULL_LORA_TARGETS}}
if [[ "${LORA_TARGET_MODULES}" != "${FULL_LORA_TARGETS}" ]]; then
    echo "Evaluation requires all seven LoRA target modules: ${FULL_LORA_TARGETS}" >&2
    exit 2
fi

if pgrep -f "trainer.experiment_name=${TRAIN_EXPERIMENT_NAME}" >/dev/null; then
    echo "Training process is still running; refusing to start evaluation." >&2
    exit 1
fi
if ! tr '\r' '\n' <"${TRAIN_LOG}" | grep -q 'Training Progress:.*100/100'; then
    echo "Training log does not show 100/100: ${TRAIN_LOG}" >&2
    exit 1
fi

read -r -a step_list <<<"${STEPS}"
actor_dirs=()
for step in "${step_list[@]}"; do
    actor_dirs+=("${CHECKPOINT_ROOT}/global_step_${step}/actor")
done
"${PYTHON_BIN}" recipe/phase2/validate_opd_checkpoints.py \
    --target-modules "${LORA_TARGET_MODULES}" \
    --expected-layers 36 \
    "${actor_dirs[@]}"

mkdir -p "${EVAL_ROOT}" "${EVAL_LOG_ROOT}" "${REPORT_ROOT}"
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

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
export MODEL_PATH=${MODEL_PATH:-${ARTIFACT_ROOT}/models/Qwen--Qwen3-4B/snapshots/master}
export TRAIN_FILE=${TRAIN_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/train.parquet}
export TEST_FILE=${TEST_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/validation.parquet}
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-${DEFAULT_TOOL_CONFIG}}

# Match the trained student architecture exactly.
export LORA_RANK=32
export LORA_ALPHA=64
export LORA_TARGET_MODULES
export LORA_LAYERED_SUMMON=False
export ROLLOUT_LOAD_FORMAT=safetensors

# Match the established Phase 1 fixed-200 protocol so results stay comparable.
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
export AGENT_TOOL_GPU_DEVICES=${AGENT_TOOL_GPU_DEVICES:-'[0]'}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-4}
export N_RESP_PER_PROMPT=1
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-3072}
export MAX_USER_TURNS=${MAX_USER_TURNS:-3}
export MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-4}
export MAX_PARALLEL_CALLS=2
export ROLLOUT_NAME=sglang
export ROLLOUT_SKIP_TOKENIZER_INIT=False
export ROLLOUT_ATTENTION_BACKEND=flashinfer
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-12288}
export ROLLOUT_ENABLE_SLEEP_MODE=False
export ROLLOUT_FREE_CACHE_ENGINE=False
export ACTOR_MODEL_DTYPE=bfloat16
export REF_MODEL_DTYPE=bfloat16
export ANSWER_LLM_JUDGE=0
export LOG_VAL_GENERATIONS=200
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_phase2_opd_sample_token_eval}

analysis_inputs=()
for step in "${step_list[@]}"; do
    eval_name="opd_sample_token_s${step}_fixed200_greedy_${EVAL_RUN_ID}"
    output_dir="${EVAL_ROOT}/s${step}"
    validation_log="${EVAL_LOG_ROOT}/s${step}.log"
    if find "${output_dir}" -maxdepth 1 -name '*.jsonl' -type f -size +0c -print -quit 2>/dev/null | grep -q .; then
        echo "Refusing to overwrite an existing evaluation dump: ${output_dir}" >&2
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

    jsonl=$(find "${output_dir}" -maxdepth 1 -name '*.jsonl' -type f -size +0c -print -quit 2>/dev/null || true)
    if [[ -z "${jsonl}" ]]; then
        echo "Evaluation finished without a non-empty JSONL dump: ${output_dir}" >&2
        exit 1
    fi
    row_count=$(wc -l <"${jsonl}")
    if [[ "${row_count}" -ne 200 ]]; then
        echo "Invalid evaluation row count for s${step}: ${row_count} != 200" >&2
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
