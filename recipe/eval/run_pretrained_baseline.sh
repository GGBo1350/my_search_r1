#!/usr/bin/env bash
# 未训练基线评测：不加载任何 checkpoint，直接用基础模型（MODEL_PATH）对
# 固定 200 条验证集跑 val_only，兼容 greedy pass@1 与采样 pass@k。
#
# 用途：
#   未训练基线 pass@1（greedy）：
#       VAL_K=1 VAL_TEMPERATURE=0 bash recipe/eval/run_pretrained_baseline.sh
#   未训练基线 pass@k（采样）：
#       VAL_K=5 VAL_TEMPERATURE=0.7 bash recipe/eval/run_pretrained_baseline.sh
#
# 必需环境变量：
#   VALIDATION_LOG         评测日志输出路径
#   VALIDATION_OUTPUT_DIR  验证轨迹输出目录（写 0.jsonl）
# 可选：
#   VAL_K                  每题采样轨迹数（默认 1）
#   VAL_TEMPERATURE        采样温度，0 表示 greedy（默认 0）
#   VAL_TOP_P              采样 top_p（默认 0.95）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

VALIDATION_LOG=${VALIDATION_LOG:?必须提供 VALIDATION_LOG}
VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR:?必须提供 VALIDATION_OUTPUT_DIR}

VAL_K=${VAL_K:-1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0}
VAL_TOP_P=${VAL_TOP_P:-0.95}
# SGLang ??????? stop ???????? tokenizer?skip_tokenizer_init=False??
ROLLOUT_SKIP_TOKENIZER_INIT=${ROLLOUT_SKIP_TOKENIZER_INIT:-True}
# SGLang attention backend?Blackwell?SM>=100???? fa3?? flashinfer?
ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-fa3}

# 温度 > 0 才真正采样；=0 时等价 greedy。
DO_SAMPLE=False
if awk "BEGIN{exit !(${VAL_TEMPERATURE} > 0)}"; then
    DO_SAMPLE=True
fi

/root/miniconda3/envs/verl/bin/ray stop --force >/dev/null 2>&1 || true
sleep 5
mkdir -p "$(dirname "${VALIDATION_LOG}")" "${VALIDATION_OUTPUT_DIR}"

export PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/verl/bin/python}
export MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen--Qwen3-4B/snapshots/master}
export TRAIN_FILE=${TRAIN_FILE:-./data/hotpotqa_v3_2k/train.parquet}
export TEST_FILE=${TEST_FILE:-./data/hotpotqa_v3_2k/validation.parquet}
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export N_RESP_PER_PROMPT=1
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-3072}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-12288}
export ACTOR_MODEL_DTYPE=bfloat16
export REF_MODEL_DTYPE=bfloat16
export VAL_BEFORE_TRAIN=True
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-200}
export ROLLOUT_DATA_ENABLED=False
export TEST_FREQ=-1
export SAVE_FREQ=-1
export RESUME_MODE=disable
export PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_lora}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_pretrained_baseline_k${VAL_K}}

exec bash "${SCRIPT_DIR}/../train_lora/run.sh" \
    trainer.val_only=True \
    trainer.validation_data_dir="${VALIDATION_OUTPUT_DIR}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_K}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${DO_SAMPLE}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
    actor_rollout_ref.rollout.skip_tokenizer_init="${ROLLOUT_SKIP_TOKENIZER_INIT}" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
    >"${VALIDATION_LOG}" 2>&1
