#!/usr/bin/env bash
# Two-GPU 931 profile for one epoch of bridge/compare OPD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if (( available_gpus < 2 )); then
    echo "This profile requires two visible GPUs; found ${available_gpus}." >&2
    exit 3
fi

export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
export STUDENT_MODEL=${STUDENT_MODEL:-/root/autodl-tmp/models/Qwen--Qwen3-4B/snapshots/master}
export TEACHER_BASE_MODEL=${TEACHER_BASE_MODEL:-${STUDENT_MODEL}}
export BRIDGE_TEACHER_ADAPTER=${BRIDGE_TEACHER_ADAPTER:-/root/autodl-tmp/models/search_r1_teacher_adapters/bridge_s75/lora_adapter}
export COMPARE_TEACHER_ADAPTER=${COMPARE_TEACHER_ADAPTER:-/root/autodl-tmp/models/search_r1_teacher_adapters/compare_s25/lora_adapter}
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
# transformer block.  Keep rank/alpha unchanged from the o_proj+down_proj run
# so this experiment isolates the effect of expanding module coverage.
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'}

export DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}
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
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
export SAVE_FREQ=${SAVE_FREQ:-100}
export TEST_FREQ=${TEST_FREQ:--1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
export RESUME_MODE=${RESUME_MODE:-disable}
export ROLLOUT_DATA_ENABLED=${ROLLOUT_DATA_ENABLED:-True}
export ROLLOUT_DATA_FREQ=${ROLLOUT_DATA_FREQ:-10}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_mopd}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_mopd_all7_lora_r32_2gpu_top32_p1024_epoch1_$(date +%Y%m%d_%H%M%S)}

echo "Student LoRA rank/alpha: ${LORA_RANK}/${LORA_ALPHA}"
echo "Student LoRA target modules: ${LORA_TARGET_MODULES}"

exec bash "${SCRIPT_DIR}/run_mopd_bridge_compare.sh" "$@"
