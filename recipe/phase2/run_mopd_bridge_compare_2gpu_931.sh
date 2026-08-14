#!/usr/bin/env bash
# Two-GPU 931 profile for routed Bridge/Comparison Forward-KL Top-32 OPD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if (( available_gpus < 2 )); then
    echo "This profile requires two visible GPUs; found ${available_gpus}." >&2
    exit 3
fi

export ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
export RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
export TEACHER_PAIR_RUN_ID=${TEACHER_PAIR_RUN_ID:-20260813_113331}
export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
export BASE_MODEL=${BASE_MODEL:-${ARTIFACT_ROOT}/models/Qwen--Qwen3-4B/snapshots/master}
export STUDENT_MODEL=${STUDENT_MODEL:-${BASE_MODEL}}
export TEACHER_BASE_MODEL=${TEACHER_BASE_MODEL:-${BASE_MODEL}}
export TEACHER_EXPORT_ROOT=${TEACHER_EXPORT_ROOT:-${ARTIFACT_ROOT}/models/search_r1_teacher_adapters/full_lora_${TEACHER_PAIR_RUN_ID}}
export BRIDGE_TEACHER_ADAPTER=${BRIDGE_TEACHER_ADAPTER:-${TEACHER_EXPORT_ROOT}/bridge_s75/lora_adapter}
export COMPARE_TEACHER_ADAPTER=${COMPARE_TEACHER_ADAPTER:-${TEACHER_EXPORT_ROOT}/compare_s25/lora_adapter}
export TRAIN_FILE=${TRAIN_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/train_opd_routed.parquet}
export TEST_FILE=${TEST_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/validation_opd_routed.parquet}

# Actor/FSDP spans both GPUs.  Student rollout uses two TP=1 replicas, while
# each static teacher is TP=2 across the same two physical GPUs.
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
export NNODES=${NNODES:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export TEACHER_TP=${TEACHER_TP:-2}
export TEACHER_NUM_REPLICAS=${TEACHER_NUM_REPLICAS:-1}
export TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-4}
export TEACHER_NNODES=${TEACHER_NNODES:-1}
export COLOCATE_TEACHERS=${COLOCATE_TEACHERS:-True}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-False}

# Apply student LoRA to every attention and MLP projection in every Qwen3
# transformer block. The shared launcher rejects any narrower target list.
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'}

export DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}
export DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-forward_kl_topk}
export USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}
export DISTILLATION_PROFILE=${DISTILLATION_PROFILE:-forward_top32}
case "${DISTILLATION_PROFILE}" in
    forward_top32) expected_loss_mode=forward_kl_topk ;;
    reverse_top32) expected_loss_mode=reverse_kl_topk ;;
    *) echo "DISTILLATION_PROFILE must be forward_top32 or reverse_top32." >&2; exit 2 ;;
esac
if [[ "${DISTILLATION_LOSS_MODE}" != "${expected_loss_mode}" || "${DISTILLATION_TOPK}" != "32" || "${USE_TASK_REWARDS}" != "False" ]]; then
    echo "Profile ${DISTILLATION_PROFILE} requires ${expected_loss_mode}/topk=32 with task rewards disabled." >&2
    exit 2
fi
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-5121}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-1024}
export MAX_USER_TURNS=${MAX_USER_TURNS:-2}
export MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-3}
export MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-2}

export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.20}
export TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.12}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
export TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-4}
export ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-True}
export ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}

export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
export AGENT_TOOL_GPU_DEVICES=${AGENT_TOOL_GPU_DEVICES:-'[0,1]'}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-4}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export SAVE_FREQ=${SAVE_FREQ:-25}
export TEST_FREQ=${TEST_FREQ:--1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-4}
export RESUME_MODE=disable
export ROLLOUT_DATA_ENABLED=${ROLLOUT_DATA_ENABLED:-True}
export ROLLOUT_DATA_FREQ=${ROLLOUT_DATA_FREQ:-10}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_mopd}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_mopd_forward_top32_all7_lora_r32_2gpu_${RUN_ID}}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${ARTIFACT_ROOT}/checkpoints/${EXPERIMENT_NAME}}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${ARTIFACT_ROOT}/rollouts/${EXPERIMENT_NAME}}
export LOG_FILE=${LOG_FILE:-${ARTIFACT_ROOT}/train_logs/${EXPERIMENT_NAME}.launch.log}
export MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB:-40}

if [[ "${TRAIN_BATCH_SIZE}" != "16" || "${N_RESP_PER_PROMPT}" != "1" || \
      "${LORA_RANK}" != "32" || "${LORA_ALPHA}" != "64" || \
      "${TOTAL_EPOCHS}" != "1" || "${TOTAL_TRAINING_STEPS}" != "100" || \
      "${SAVE_FREQ}" != "25" || "${MAX_ACTOR_CKPT_TO_KEEP}" != "4" ]]; then
    echo "This 931 comparison profile is fixed to batch=16, n=1, LoRA r32/a64, 100 steps, and checkpoints at 25/50/75/100." >&2
    exit 2
fi
if ! [[ "${MIN_FREE_DISK_GB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MIN_FREE_DISK_GB must be a positive integer; got ${MIN_FREE_DISK_GB}." >&2
    exit 2
fi

available_disk_kb=$(df -Pk "${ARTIFACT_ROOT}" | awk 'NR == 2 {print $4}')
required_disk_kb=$((MIN_FREE_DISK_GB * 1024 * 1024))
if [[ -z "${available_disk_kb}" || "${available_disk_kb}" -lt "${required_disk_kb}" ]]; then
    echo "Insufficient free space under ${ARTIFACT_ROOT}: need at least ${MIN_FREE_DISK_GB}GB before saving four approximately 9GB checkpoints." >&2
    df -h "${ARTIFACT_ROOT}" >&2 || true
    exit 5
fi

for d in "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}"; do
    if [[ -d "${d}" ]] && find "${d}" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing to reuse a non-empty experiment path: ${d}" >&2
        exit 4
    fi
done
if [[ -e "${LOG_FILE}" ]]; then
    echo "Refusing to append to an existing experiment log: ${LOG_FILE}" >&2
    exit 4
fi

mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
BASE_MODEL="${TEACHER_BASE_MODEL}" \
TEACHER_PAIR_RUN_ID="${TEACHER_PAIR_RUN_ID}" \
TEACHER_EXPORT_ROOT="${TEACHER_EXPORT_ROOT}" \
BRIDGE_TEACHER_ADAPTER="${BRIDGE_TEACHER_ADAPTER}" \
COMPARE_TEACHER_ADAPTER="${COMPARE_TEACHER_ADAPTER}" \
    bash "${SCRIPT_DIR}/prepare_full_lora_teacher_pair_931.sh"

echo "Student LoRA rank/alpha: ${LORA_RANK}/${LORA_ALPHA}"
echo "Student LoRA target modules: ${LORA_TARGET_MODULES}"
echo "DISTILLATION_PROFILE=${DISTILLATION_PROFILE} DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE} DISTILLATION_TOPK=${DISTILLATION_TOPK}"
echo "USE_TASK_REWARDS=${USE_TASK_REWARDS} USE_POLICY_GRADIENT=False"
echo "BRIDGE_TEACHER_ADAPTER=${BRIDGE_TEACHER_ADAPTER}"
echo "COMPARE_TEACHER_ADAPTER=${COMPARE_TEACHER_ADAPTER}"
echo "CHECKPOINT_STEPS=25,50,75,100 MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP}"
echo "MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

exec bash "${SCRIPT_DIR}/run_mopd_bridge_compare.sh" "$@"
