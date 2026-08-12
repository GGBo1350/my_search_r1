#!/bin/bash
# Search-R1 V3：Qwen 4B + LoRA + GRPO 训练配置。
#
# 默认按单张 96G 或两张 48G 的保守起步参数设置：每个问题采样 8 条轨迹，
# 但只使用 8 个问题组成一个 Prompt batch，并限制 vLLM 同时处理的序列数。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA。alpha 默认跟随 rank 取两倍；也可以分别通过环境变量覆盖。
LORA_RANK=${LORA_RANK:-32}
if ! [[ "${LORA_RANK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LORA_RANK 必须是正整数，当前值：${LORA_RANK}" >&2
    exit 2
fi
LORA_ALPHA=${LORA_ALPHA:-$((LORA_RANK * 2))}
if ! [[ "${LORA_ALPHA}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LORA_ALPHA 必须是正整数，当前值：${LORA_ALPHA}" >&2
    exit 2
fi
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
# 当前单卡 FSDP + vLLM 路径使用非分层收集，避免 rollout 侧 LoRA
# 参数不完整时静默产生异常输出。
LORA_LAYERED_SUMMON=${LORA_LAYERED_SUMMON:-False}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-safetensors}

# 模型和采样。n=8 表示每个问题生成 8 条候选轨迹，不表示调用 8 次搜索。
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
export TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3_2k/train.parquet}
export TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3_2k/validation.parquet}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-3072}

# 小 Prompt batch 控制单步生成量。verl 会按 rollout.n 扩展实际轨迹数量：
# 默认每步为 8 个问题 x 8 条轨迹 = 64 条轨迹。
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}

# 默认以单张 96G 为基线。两张 48G 可设置 NGPUS_PER_NODE=2；LoRA 下 4B
# 通常不需要为了装下模型而设置 TP=2，TP=1 可以减少小模型的跨卡通信。
export NNODES=${NNODES:-1}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-12288}
export ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-False}
export ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
export ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-bfloat16}
export REF_MODEL_DTYPE=${REF_MODEL_DTYPE:-bfloat16}
export ACTOR_USE_TORCH_COMPILE=${ACTOR_USE_TORCH_COMPILE:-False}

# 96G/两张 48G 默认不卸载，以免 CPU 传输拖慢训练。32G 或单张 48G 显存不足时，
# 可在启动命令里把下面三个变量覆盖为 True。
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}

# 3e-5 在 n=8、batch_size=8 的长跑中更新过快；默认降为 5e-6，并适度
# 增强 KL 约束。两个参数仍可通过环境变量在启动时覆盖。
export ACTOR_LR=${ACTOR_LR:-5e-6}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.003}
export PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_lora}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_lora_r${LORA_RANK}_a${LORA_ALPHA}_n${N_RESP_PER_PROMPT}}

exec bash "${SCRIPT_DIR}/run.sh" \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}" \
    actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
    actor_rollout_ref.rollout.layered_summon="${LORA_LAYERED_SUMMON}" \
    "$@"
