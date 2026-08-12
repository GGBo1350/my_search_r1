#!/bin/bash
# Search-R1 v3: HotpotQA local search with parallel comparison calls and
# sequential bridge calls.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Model and data
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3/train.parquet}
TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3/validation.parquet}

# Algorithm
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}

# Multi-turn agent. One <tool_calls> block may contain two search calls.
MAX_USER_TURNS=${MAX_USER_TURNS:-3}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-4}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-2}
TOOL_FORMAT=${TOOL_FORMAT:-search_r1_v3}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-6144}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-bfloat16}
REF_MODEL_DTYPE=${REF_MODEL_DTYPE:-bfloat16}
ACTOR_USE_TORCH_COMPILE=${ACTOR_USE_TORCH_COMPILE:-False}

# Training
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}

# Rollout
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-null}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-False}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-8}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-null}

# Memory controls. Small single-GPU runs can enable CPU offload through these
# variables without duplicating the full launch command.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}
ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-True}

# V3 reward and local tool. The YAML currently exposes only `search`, while
# keeping verl's generic registry so future tools can be added there.
REWARD_PATH=${REWARD_PATH:-recipe/core/my_reward.py}
REWARD_NAME=${REWARD_NAME:-compute_score}
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-recipe/core/tool_config_hybrid.yaml}

# Logging/checkpoints
PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_hotpotqa_search_grpo}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-10}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-null}
RESUME_MODE=${RESUME_MODE:-auto}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-10}

# Rollout 样本导出默认关闭。开启后每隔指定 step 保存一次 JSONL，便于人工
# review 问题、标准答案、模型完整输出和奖励明细。
ROLLOUT_DATA_ENABLED=${ROLLOUT_DATA_ENABLED:-False}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-./outputs/rollouts}
ROLLOUT_DATA_FREQ=${ROLLOUT_DATA_FREQ:-5}
if ! [[ "${ROLLOUT_DATA_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROLLOUT_DATA_FREQ 必须是正整数，当前值：${ROLLOUT_DATA_FREQ}" >&2
    exit 2
fi
case "${ROLLOUT_DATA_ENABLED,,}" in
    true|1|yes)
        TRAINER_ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR}
        ;;
    false|0|no)
        TRAINER_ROLLOUT_DATA_DIR=null
        ;;
    *)
        echo "ROLLOUT_DATA_ENABLED 必须是 True/False，当前值：${ROLLOUT_DATA_ENABLED}" >&2
        exit 2
        ;;
esac

# AutoDL 上的 SwanLab 密钥保存在仓库之外。其他机器没有该文件时直接跳过，
# 仍可通过环境变量 SWANLAB_API_KEY、SWANLAB_MODE 和 SWANLAB_LOG_DIR 配置。
SWANLAB_ENV_FILE=${SWANLAB_ENV_FILE:-/root/autodl-tmp/.secrets/swanlab.env}
if [[ -r "${SWANLAB_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${SWANLAB_ENV_FILE}"
    set +a
fi
# 当前仓库的 V1 trainer 依赖未随 setup.py 安装的 transfer_queue。
# V3 使用已支持多轮工具调用的 legacy trainer；依赖补齐后可显式切回 True。
TRAINER_USE_V1=${TRAINER_USE_V1:-False}

ACTOR_LR=${ACTOR_LR:-5e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.003}
PYTHON_BIN=${PYTHON_BIN:-python}

# verl v0.8.0 没有 trainer.use_v1 字段；较新版本则需要显式选择
# legacy trainer。按配置文件自动决定是否传入该覆盖项，让同一份 V3
# 脚本兼容 Torch 2.8/vLLM 0.11 环境和当前开发环境。
TRAINER_COMPAT_ARGS=()
if grep -qE '^[[:space:]]+use_v1:' "${REPO_ROOT}/verl/trainer/config/ppo_trainer.yaml"; then
    TRAINER_COMPAT_ARGS+=("trainer.use_v1=${TRAINER_USE_V1}")
fi

# pip/uv 安装的 CUDA wheel 会把动态库放在 site-packages/nvidia/*/lib。
# 部分容器不会自动把这些目录加入动态链接器搜索路径，导致 vLLM 的
# cuMem allocator 找不到 libnvrtc.so。只在目录存在时追加，系统 CUDA
# 环境不会受到影响。
PY_SITE_PACKAGES=$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')
NVIDIA_WHEEL_ROOT="${PY_SITE_PACKAGES}/nvidia"
if [[ -d "${NVIDIA_WHEEL_ROOT}" ]]; then
    CUDA_WHEEL_LIBS=$(find "${NVIDIA_WHEEL_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib | paste -sd: -)
    if [[ -n "${CUDA_WHEEL_LIBS}" ]]; then
        export LD_LIBRARY_PATH="${CUDA_WHEEL_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.return_raw_chat=True \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss="${ACTOR_USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
    actor_rollout_ref.ref.fsdp_config.model_dtype="${REF_MODEL_DTYPE}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    +actor_rollout_ref.rollout.enable_sleep_mode="${ROLLOUT_ENABLE_SLEEP_MODE}" \
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format="${TOOL_FORMAT}" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls="${MAX_PARALLEL_CALLS}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    reward.num_workers="${REWARD_NUM_WORKERS}" \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name="${REWARD_NAME}" \
    trainer.logger="${TRAINER_LOGGER}" \
    "${TRAINER_COMPAT_ARGS[@]}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
    trainer.rollout_data_dir="${TRAINER_ROLLOUT_DATA_DIR}" \
    +trainer.rollout_data_freq="${ROLLOUT_DATA_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.resume_mode="${RESUME_MODE}" \
    ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
    "$@"
