#!/usr/bin/env bash
# Reusable two-GPU, single-teacher Sample-K3 OPD stage for 931.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE_NAME=${STAGE_NAME:?set STAGE_NAME to compare or bridge}
TEACHER_ADAPTER=${TEACHER_ADAPTER:?set TEACHER_ADAPTER}
TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE}
TEST_FILE=${TEST_FILE:?set TEST_FILE}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?set EXPERIMENT_NAME}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR}

ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/autodl-tmp}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
BASE_MODEL=${BASE_MODEL:-${ARTIFACT_ROOT}/models/Qwen--Qwen3-4B/snapshots/master}
STUDENT_MODEL=${STUDENT_MODEL:-${BASE_MODEL}}
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${ARTIFACT_ROOT}/rollouts/${EXPERIMENT_NAME}}
LOG_FILE=${LOG_FILE:-${ARTIFACT_ROOT}/train_logs/${EXPERIMENT_NAME}.launch.log}
FULL_LORA_TARGETS='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'

case "${STAGE_NAME}" in compare|bridge) ;; *) echo "STAGE_NAME must be compare or bridge." >&2; exit 2 ;; esac
if [[ ! -x "${PYTHON_BIN}" || ! -f "${BASE_MODEL}/config.json" ]]; then
    echo "Python or Qwen3-4B Base is missing: ${PYTHON_BIN} / ${BASE_MODEL}" >&2
    exit 3
fi
if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" ]]; then
    echo "Training data is missing: ${TRAIN_FILE} / ${TEST_FILE}" >&2
    exit 3
fi
if [[ ! -s "${TEACHER_ADAPTER}/adapter_config.json" || ! -s "${TEACHER_ADAPTER}/adapter_model.safetensors" ]]; then
    echo "Teacher adapter is incomplete: ${TEACHER_ADAPTER}" >&2
    exit 3
fi
available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if (( available_gpus < 2 )); then
    echo "This Sample-K3 stage requires two visible GPUs; found ${available_gpus}." >&2
    exit 3
fi
if [[ -d "${CHECKPOINT_DIR}" ]] && find "${CHECKPOINT_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to reuse a non-empty checkpoint directory: ${CHECKPOINT_DIR}" >&2
    exit 4
fi
if [[ -d "${ROLLOUT_DATA_DIR}" ]] && find "${ROLLOUT_DATA_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to reuse a non-empty rollout directory: ${ROLLOUT_DATA_DIR}" >&2
    exit 4
fi

"${PYTHON_BIN}" recipe/phase2/verify_teacher_adapters.py \
    --rank 32 --target-modules "${FULL_LORA_TARGETS}" "${TEACHER_ADAPTER}"

if [[ -f recipe/train_lora/run_lora.sh ]]; then
    TRAIN_ENTRY=recipe/train_lora/run_lora.sh
elif [[ -f recipe/v3/run_lora.sh ]]; then
    TRAIN_ENTRY=recipe/v3/run_lora.sh
else
    echo "Cannot find the LoRA training entrypoint." >&2
    exit 3
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_PATH:-}" ]]; then
    if [[ ! -d "${RESUME_FROM_PATH}/actor" ]]; then
        echo "Resume actor checkpoint is missing: ${RESUME_FROM_PATH}/actor" >&2
        exit 4
    fi
    if [[ -e "${RESUME_FROM_PATH}/data.pt" ]]; then
        echo "Serial stage handoff must not include a dataloader state: ${RESUME_FROM_PATH}/data.pt" >&2
        exit 4
    fi
    RESUME_MODE=resume_path
    RESUME_ARGS+=(trainer.resume_from_path="${RESUME_FROM_PATH}")
else
    RESUME_MODE=disable
fi

mkdir -p "$(dirname "${LOG_FILE}")" "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/cuda/bin:${PATH}"
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MODEL_PATH="${STUDENT_MODEL}"
export TRAIN_FILE TEST_FILE PYTHON_BIN TOOL_CONFIG_PATH
export NGPUS_PER_NODE=2 NNODES=1 ROLLOUT_TP=1
export LORA_RANK=32 LORA_ALPHA=64 LORA_TARGET_MODULES="${FULL_LORA_TARGETS}"
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
export ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-flashinfer}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.20}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
export ROLLOUT_ENABLE_SLEEP_MODE=True
export ROLLOUT_FREE_CACHE_ENGINE=True

export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
export AGENT_TOOL_GPU_DEVICES=${AGENT_TOOL_GPU_DEVICES:-'[0,1]'}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-4}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# Serial stages use one shared 100-step optimizer/LR-scheduler horizon. The
# curriculum launcher chooses total_epochs for each stage according to its
# dataset length and resume step.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export SAVE_FREQ=${SAVE_FREQ:-25}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
export TEST_FREQ=-1 VAL_BEFORE_TRAIN=False RESUME_MODE
export ROLLOUT_DATA_ENABLED=True
export ROLLOUT_DATA_FREQ=${ROLLOUT_DATA_FREQ:-10}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_opd_serial_k3}
export EXPERIMENT_NAME CHECKPOINT_DIR ROLLOUT_DATA_DIR
export ANSWER_LLM_JUDGE=0

TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.12}
TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-4}

echo "STAGE_NAME=${STAGE_NAME}"
echo "DISTILLATION_LOSS_MODE=k3 USE_TASK_REWARDS=False USE_POLICY_GRADIENT=False"
echo "TEACHER_TP=1 TEACHER_NUM_REPLICAS=2"
echo "TEACHER_ADAPTER=${TEACHER_ADAPTER}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "RESUME_MODE=${RESUME_MODE} RESUME_FROM_PATH=${RESUME_FROM_PATH:-none}"
echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} SAVE_FREQ=${SAVE_FREQ}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

exec bash "${TRAIN_ENTRY}" \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
    +actor_rollout_ref.rollout.agent.tool_gpu_devices="${AGENT_TOOL_GPU_DEVICES}" \
    distillation.enabled=True \
    distillation.n_gpus_per_node=2 \
    distillation.nnodes=1 \
    distillation.colocate_with_actor=True \
    distillation.teacher_key=teacher_route \
    distillation.teacher_models.teacher_model.model_path="${BASE_MODEL}" \
    distillation.teacher_models.teacher_model.lora_adapter_path="${TEACHER_ADAPTER}" \
    distillation.teacher_models.teacher_model.lora_rank=32 \
    distillation.teacher_models.teacher_model.lora_target_modules="${FULL_LORA_TARGETS}" \
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
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.sglang.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
    distillation.distillation_loss.loss_mode=k3 \
    distillation.distillation_loss.topk=null \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.distillation_loss_coef=1.0 \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    distillation.distillation_loss.use_policy_gradient=False \
    trainer.del_local_ckpt_after_load=False \
    "${RESUME_ARGS[@]}" \
    "$@"
