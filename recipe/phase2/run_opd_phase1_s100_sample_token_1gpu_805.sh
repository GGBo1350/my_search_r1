#!/usr/bin/env bash
# Phase 2: single-teacher OPD on one 96 GB GPU.
#
# Teacher: Phase 1 full-module LoRA global_step_100.
# Signal: sampled-token k3 estimator only (teacher top-k is disabled).
# Student: rank-32 LoRA on q/k/v/o and gate/up/down in every Qwen3 block.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
BASE_MODEL=${BASE_MODEL:-${ARTIFACT_ROOT}/models/Qwen--Qwen3-4B/snapshots/master}
PHASE1_RUN=${PHASE1_RUN:-qwen3_4b_phase1_full_lora_125step_939_20260811_104813}
TEACHER_ACTOR_CHECKPOINT=${TEACHER_ACTOR_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/${PHASE1_RUN}/global_step_100/actor}
TEACHER_EXPORT_ROOT=${TEACHER_EXPORT_ROOT:-${ARTIFACT_ROOT}/models/search_r1_teacher_adapters/phase1_full_lora_s100}
TEACHER_ADAPTER=${TEACHER_ADAPTER:-${TEACHER_EXPORT_ROOT}/lora_adapter}

# A single teacher does not require teacher_route.  Use the Phase 1 dataset
# already present on 805; every sample is routed to the sole teacher.
TRAIN_FILE=${TRAIN_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/train.parquet}
TEST_FILE=${TEST_FILE:-${REPO_ROOT}/data/hotpotqa_v3_hard_1600/validation.parquet}

FULL_LORA_TARGETS='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-${FULL_LORA_TARGETS}}
TEACHER_LORA_RANK=${TEACHER_LORA_RANK:-32}
TEACHER_LORA_TARGET_MODULES=${TEACHER_LORA_TARGET_MODULES:-${FULL_LORA_TARGETS}}

# Fail closed: this experiment must not silently fall back to the historical
# o_proj/down_proj-only configuration.
if [[ "${LORA_TARGET_MODULES}" != "${FULL_LORA_TARGETS}" ]]; then
    echo "Student LoRA must cover all seven Qwen3 projection types: ${FULL_LORA_TARGETS}" >&2
    exit 2
fi
if [[ "${TEACHER_LORA_TARGET_MODULES}" != "${FULL_LORA_TARGETS}" ]]; then
    echo "Teacher LoRA must cover all seven Qwen3 projection types: ${FULL_LORA_TARGETS}" >&2
    exit 2
fi

# k3 is a single-sample reverse-KL estimator.  Since this mode does not use
# teacher top-k logprobs, the teacher returns only the logprob of each token
# sampled by the student.  Keep policy-gradient distillation disabled so the
# signal is backpropagated directly, matching the stable sample-token profile.
DISTILLATION_LOSS_MODE=k3
DISTILLATION_TOPK=null
USE_POLICY_GRADIENT=False
USE_TASK_REWARDS=False

if [[ -f recipe/train_lora/run_lora.sh ]]; then
    TRAIN_ENTRY=recipe/train_lora/run_lora.sh
    TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}
elif [[ -f recipe/v3/run_lora.sh ]]; then
    TRAIN_ENTRY=recipe/v3/run_lora.sh
    TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/v3/tool_config_hybrid.yaml}
else
    echo "Cannot find the LoRA training entrypoint." >&2
    exit 3
fi

if [[ ! -f recipe/phase2/extract_teacher_lora.py || ! -f recipe/phase2/verify_teacher_adapters.py ]]; then
    echo "Missing Phase 2 teacher adapter preparation scripts under recipe/phase2." >&2
    exit 3
fi

export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/cuda/bin:${PATH}"
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_opd_phase1_s100_sample_token_k3_all7_lora_r32_1gpu_${RUN_ID}}
PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_opd_sample_token}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${ARTIFACT_ROOT}/checkpoints/${EXPERIMENT_NAME}}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${ARTIFACT_ROOT}/rollouts/${EXPERIMENT_NAME}}
LOG_FILE=${LOG_FILE:-${ARTIFACT_ROOT}/train_logs/${EXPERIMENT_NAME}.launch.log}

echo "RUN_ID=${RUN_ID}"
echo "TEACHER_ACTOR_CHECKPOINT=${TEACHER_ACTOR_CHECKPOINT}"
echo "TEACHER_ADAPTER=${TEACHER_ADAPTER}"
echo "DISTILLATION_SIGNAL=sample_token"
echo "DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE} DISTILLATION_TOPK=${DISTILLATION_TOPK}"
echo "STUDENT_LORA=${LORA_TARGET_MODULES} rank=${LORA_RANK} alpha=${LORA_ALPHA}"
echo "TEACHER_LORA=${TEACHER_LORA_TARGET_MODULES} rank=${TEACHER_LORA_RANK}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

case "${OPD_CONFIG_ONLY:-False}" in
    True|true|1|yes)
        echo "Configuration-only check complete; adapter extraction and training were not started."
        exit 0
        ;;
    False|false|0|no) ;;
    *) echo "OPD_CONFIG_ONLY must be True or False." >&2; exit 2 ;;
esac

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "No NVIDIA GPU is attached." >&2
    exit 4
fi
if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
    echo "Base model is incomplete: ${BASE_MODEL}" >&2
    exit 4
fi
if [[ ! -d "${TEACHER_ACTOR_CHECKPOINT}" ]]; then
    echo "Teacher actor checkpoint is missing: ${TEACHER_ACTOR_CHECKPOINT}" >&2
    exit 4
fi
if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" ]]; then
    echo "OPD train/validation data is missing: ${TRAIN_FILE} / ${TEST_FILE}" >&2
    exit 4
fi

# Export the small PEFT adapter once.  The extractor refuses to overwrite a
# non-empty directory, so a partial prior export fails safely.
if [[ ! -f "${TEACHER_ADAPTER}/adapter_config.json" || ! -f "${TEACHER_ADAPTER}/adapter_model.safetensors" ]]; then
    "${PYTHON_BIN}" recipe/phase2/extract_teacher_lora.py \
        --actor-checkpoint "${TEACHER_ACTOR_CHECKPOINT}" \
        --base-model "${BASE_MODEL}" \
        --load-device cpu \
        --output "${TEACHER_EXPORT_ROOT}"
fi

"${PYTHON_BIN}" recipe/phase2/verify_teacher_adapters.py \
    --rank "${TEACHER_LORA_RANK}" \
    --target-modules "${TEACHER_LORA_TARGET_MODULES}" \
    "${TEACHER_ADAPTER}"

mkdir -p "$(dirname "${LOG_FILE}")" "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}"
exec >"${LOG_FILE}" 2>&1

export MODEL_PATH=${STUDENT_MODEL:-${BASE_MODEL}}
export TRAIN_FILE TEST_FILE PYTHON_BIN TOOL_CONFIG_PATH
export NGPUS_PER_NODE=1
export NNODES=1
export ROLLOUT_TP=1

export LORA_RANK LORA_ALPHA LORA_TARGET_MODULES
export ACTOR_LR=${ACTOR_LR:-5e-6}
export ACTOR_USE_KL_LOSS=False
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export N_RESP_PER_PROMPT=1
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-5121}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-1024}
export MAX_USER_TURNS=${MAX_USER_TURNS:-2}
export MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-3}
export MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-2}

export ROLLOUT_NAME=sglang
export ROLLOUT_SKIP_TOKENIZER_INIT=False
export ROLLOUT_ATTENTION_BACKEND=flashinfer
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.20}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
export ROLLOUT_ENABLE_SLEEP_MODE=True
export ROLLOUT_FREE_CACHE_ENGINE=True

# Two retrieval workers were stable in the Phase 1 evaluation.  Each worker
# loads a retrieval model on GPU, so keep this conservative for colocated OPD.
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
export AGENT_TOOL_GPU_DEVICES=${AGENT_TOOL_GPU_DEVICES:-'[0]'}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-2}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}

export TOTAL_EPOCHS=1
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export SAVE_FREQ=${SAVE_FREQ:-25}
export MAX_ACTOR_CKPT_TO_KEEP=4
export TEST_FREQ=-1
export VAL_BEFORE_TRAIN=False
export RESUME_MODE=disable
export ROLLOUT_DATA_ENABLED=True
export ROLLOUT_DATA_FREQ=${ROLLOUT_DATA_FREQ:-10}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}
export PROJECT_NAME EXPERIMENT_NAME CHECKPOINT_DIR ROLLOUT_DATA_DIR
export ANSWER_LLM_JUDGE=0

TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.12}
TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-4}

echo "TRAIN_ENTRY=${TRAIN_ENTRY}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} SAVE_FREQ=${SAVE_FREQ}"
echo "ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL} TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL}"
echo "AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS}"

exec bash "${TRAIN_ENTRY}" \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.agent.tool_gpu_devices="${AGENT_TOOL_GPU_DEVICES}" \
    distillation.enabled=True \
    distillation.n_gpus_per_node=1 \
    distillation.nnodes=1 \
    distillation.colocate_with_actor=True \
    distillation.teacher_key=teacher_route \
    distillation.teacher_models.teacher_model.model_path="${BASE_MODEL}" \
    distillation.teacher_models.teacher_model.lora_adapter_path="${TEACHER_ADAPTER}" \
    distillation.teacher_models.teacher_model.lora_rank="${TEACHER_LORA_RANK}" \
    distillation.teacher_models.teacher_model.lora_target_modules="${TEACHER_LORA_TARGET_MODULES}" \
    distillation.teacher_models.teacher_model.inference.name=sglang \
    distillation.teacher_models.teacher_model.inference.dtype=bfloat16 \
    distillation.teacher_models.teacher_model.inference.temperature=1.0 \
    distillation.teacher_models.teacher_model.inference.enforce_eager=True \
    distillation.teacher_models.teacher_model.inference.skip_tokenizer_init=False \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL}" \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=8192 \
    distillation.teacher_models.teacher_model.inference.max_num_seqs="${TEACHER_MAX_NUM_SEQS}" \
    distillation.teacher_models.teacher_model.inference.free_cache_engine=True \
    distillation.teacher_models.teacher_model.inference.load_format=auto \
    distillation.teacher_models.teacher_model.inference.prompt_length="${MAX_PROMPT_LENGTH}" \
    distillation.teacher_models.teacher_model.inference.response_length="${MAX_RESPONSE_LENGTH}" \
    distillation.teacher_models.teacher_model.inference.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.sglang.attention_backend=flashinfer \
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}" \
    distillation.distillation_loss.topk="${DISTILLATION_TOPK}" \
    distillation.distillation_loss.use_task_rewards="${USE_TASK_REWARDS}" \
    distillation.distillation_loss.distillation_loss_coef=1.0 \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    distillation.distillation_loss.use_policy_gradient="${USE_POLICY_GRADIENT}" \
    distillation.distillation_loss.policy_loss_mode=vanilla \
    "$@"
