#!/usr/bin/env bash
# Phase 2: native verl v0.8 multi-teacher OPD for Search-R1.
# bridge samples route to the s75 teacher; compare samples route to the s25 teacher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STUDENT_MODEL=${STUDENT_MODEL:?must provide a Hugging Face student model path}
TEACHER_BASE_MODEL=${TEACHER_BASE_MODEL:?must provide the Qwen3-4B teacher base model path}
BRIDGE_TEACHER_ADAPTER=${BRIDGE_TEACHER_ADAPTER:?must provide bridge teacher s75 LoRA adapter path}
COMPARE_TEACHER_ADAPTER=${COMPARE_TEACHER_ADAPTER:?must provide compare teacher s25 LoRA adapter path}
TRAIN_FILE=${TRAIN_FILE:?must provide routed OPD train Parquet}
TEST_FILE=${TEST_FILE:?must provide validation Parquet}

PYTHON_BIN=${PYTHON_BIN:-python}
PYTHON_ENV_BIN=$(dirname "${PYTHON_BIN}")
export PATH="${PYTHON_ENV_BIN}:/usr/local/cuda/bin:${PATH}"
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-2}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_NUM_REPLICAS=${TEACHER_NUM_REPLICAS:-1}
TEACHER_LORA_RANK=${TEACHER_LORA_RANK:-32}
TEACHER_LORA_TARGET_MODULES=${TEACHER_LORA_TARGET_MODULES:-"[o_proj,down_proj]"}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-16}
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-forward_kl_topk}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}
DISTILLATION_LOSS_COEF=${DISTILLATION_LOSS_COEF:-1.0}
# Single-GPU OPD keeps three SGLang engines beside the actor.  Leave enough
# headroom for the 2048 + 4096 token actor backward pass instead of filling the
# card with persistent KV caches.
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.12}
TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-4}
TEACHER_INFERENCE_BACKEND=${TEACHER_INFERENCE_BACKEND:-sglang}
COLOCATE_TEACHERS=${COLOCATE_TEACHERS:-True}
SKIP_GPU_PREFLIGHT=${SKIP_GPU_PREFLIGHT:-False}

expected_teacher_gpus=$((2 * TEACHER_NUM_REPLICAS * TEACHER_TP))
configured_teacher_gpus=$((TEACHER_NGPUS_PER_NODE * TEACHER_NNODES))
if (( expected_teacher_gpus != configured_teacher_gpus )); then
    echo "teacher pool mismatch: two teachers require ${expected_teacher_gpus} GPUs, configured ${configured_teacher_gpus}" >&2
    exit 2
fi

"${PYTHON_BIN}" recipe/phase2/verify_opd_routes.py "${TRAIN_FILE}"
"${PYTHON_BIN}" recipe/phase2/verify_teacher_adapters.py \
    --rank "${TEACHER_LORA_RANK}" \
    --target-modules "${TEACHER_LORA_TARGET_MODULES}" \
    "${BRIDGE_TEACHER_ADAPTER}" \
    "${COMPARE_TEACHER_ADAPTER}"

case "${SKIP_GPU_PREFLIGHT,,}" in
    false|0|no)
        available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
        case "${COLOCATE_TEACHERS,,}" in
            true|1|yes) required_gpus=$((NGPUS_PER_NODE * NNODES)) ;;
            false|0|no) required_gpus=$((NGPUS_PER_NODE * NNODES + configured_teacher_gpus)) ;;
            *) echo "COLOCATE_TEACHERS must be True/False" >&2; exit 2 ;;
        esac
        if (( available_gpus < required_gpus )); then
            echo "multi-teacher OPD requires ${required_gpus} visible GPUs with COLOCATE_TEACHERS=${COLOCATE_TEACHERS}; found ${available_gpus}" >&2
            echo "Do not bypass this check unless Ray spans additional nodes." >&2
            exit 3
        fi
        ;;
    true|1|yes) ;;
    *) echo "SKIP_GPU_PREFLIGHT must be True/False" >&2; exit 2 ;;
esac

export MODEL_PATH="${STUDENT_MODEL}"
export TRAIN_FILE TEST_FILE PYTHON_BIN NGPUS_PER_NODE NNODES
export ROLLOUT_NAME=${ROLLOUT_NAME:-sglang}
export ROLLOUT_SKIP_TOKENIZER_INIT=${ROLLOUT_SKIP_TOKENIZER_INIT:-False}
export ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-flashinfer}
# PEFT builds the actor from the Hugging Face Qwen3 module tree, whose
# projections are not fused.  SGLang accepts these names and maps q/k/v and
# gate/up to its packed qkv_proj and gate_up_proj rollout modules.
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.20}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
export ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-True}
export ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
export TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}
export AGENT_TOOL_GPU_DEVICES=${AGENT_TOOL_GPU_DEVICES:-null}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))}
export ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-False}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:--1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_mopd}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_mopd_bridge_s75_compare_s25}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/autodl-tmp/checkpoints/${EXPERIMENT_NAME}}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-/root/autodl-tmp/rollouts/${EXPERIMENT_NAME}}

exec bash recipe/train_lora/run_lora.sh \
    actor_rollout_ref.rollout.skip_tokenizer_init="${ROLLOUT_SKIP_TOKENIZER_INIT}" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
    actor_rollout_ref.rollout.agent.tool_gpu_devices="${AGENT_TOOL_GPU_DEVICES}" \
    distillation.enabled=True \
    distillation.n_gpus_per_node="${TEACHER_NGPUS_PER_NODE}" \
    distillation.nnodes="${TEACHER_NNODES}" \
    distillation.teacher_key=teacher_route \
    distillation.colocate_with_actor="${COLOCATE_TEACHERS}" \
    +distillation.teacher_models.bridge.key=bridge \
    +distillation.teacher_models.bridge.model_path="${TEACHER_BASE_MODEL}" \
    +distillation.teacher_models.bridge.lora_adapter_path="${BRIDGE_TEACHER_ADAPTER}" \
    +distillation.teacher_models.bridge.lora_rank="${TEACHER_LORA_RANK}" \
    +distillation.teacher_models.bridge.lora_target_modules="${TEACHER_LORA_TARGET_MODULES}" \
    +distillation.teacher_models.bridge.num_replicas="${TEACHER_NUM_REPLICAS}" \
    +distillation.teacher_models.bridge.inference.name="${TEACHER_INFERENCE_BACKEND}" \
    +distillation.teacher_models.bridge.inference.dtype=bfloat16 \
    +distillation.teacher_models.bridge.inference.enforce_eager=True \
    +distillation.teacher_models.bridge.inference.skip_tokenizer_init=False \
    +distillation.teacher_models.bridge.inference.tensor_model_parallel_size="${TEACHER_TP}" \
    +distillation.teacher_models.bridge.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL}" \
    +distillation.teacher_models.bridge.inference.max_num_seqs="${TEACHER_MAX_NUM_SEQS}" \
    +distillation.teacher_models.bridge.inference.load_format=auto \
    +distillation.teacher_models.bridge.inference.prompt_length="${MAX_PROMPT_LENGTH}" \
    +distillation.teacher_models.bridge.inference.response_length="${MAX_RESPONSE_LENGTH}" \
    +distillation.teacher_models.bridge.inference.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-12288}" \
    +distillation.teacher_models.bridge.inference.engine_kwargs.sglang.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
    +distillation.teacher_models.compare.key=compare \
    +distillation.teacher_models.compare.model_path="${TEACHER_BASE_MODEL}" \
    +distillation.teacher_models.compare.lora_adapter_path="${COMPARE_TEACHER_ADAPTER}" \
    +distillation.teacher_models.compare.lora_rank="${TEACHER_LORA_RANK}" \
    +distillation.teacher_models.compare.lora_target_modules="${TEACHER_LORA_TARGET_MODULES}" \
    +distillation.teacher_models.compare.num_replicas="${TEACHER_NUM_REPLICAS}" \
    +distillation.teacher_models.compare.inference.name="${TEACHER_INFERENCE_BACKEND}" \
    +distillation.teacher_models.compare.inference.dtype=bfloat16 \
    +distillation.teacher_models.compare.inference.enforce_eager=True \
    +distillation.teacher_models.compare.inference.skip_tokenizer_init=False \
    +distillation.teacher_models.compare.inference.tensor_model_parallel_size="${TEACHER_TP}" \
    +distillation.teacher_models.compare.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL}" \
    +distillation.teacher_models.compare.inference.max_num_seqs="${TEACHER_MAX_NUM_SEQS}" \
    +distillation.teacher_models.compare.inference.load_format=auto \
    +distillation.teacher_models.compare.inference.prompt_length="${MAX_PROMPT_LENGTH}" \
    +distillation.teacher_models.compare.inference.response_length="${MAX_RESPONSE_LENGTH}" \
    +distillation.teacher_models.compare.inference.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-12288}" \
    +distillation.teacher_models.compare.inference.engine_kwargs.sglang.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}" \
    distillation.distillation_loss.topk="${DISTILLATION_TOPK}" \
    distillation.distillation_loss.use_task_rewards="${USE_TASK_REWARDS}" \
    distillation.distillation_loss.distillation_loss_coef="${DISTILLATION_LOSS_COEF}" \
    distillation.distillation_loss.use_policy_gradient=False \
    "$@"
