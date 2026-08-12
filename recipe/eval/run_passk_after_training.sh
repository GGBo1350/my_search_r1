#!/usr/bin/env bash
# 等待训练结束后，加载最终 checkpoint，对固定 200 条验证集做 pass@k 评测。
# 与 run_fixed200_after_training.sh 的区别：
#   - 每个问题采样 VAL_K 条轨迹（val_kwargs.n=VAL_K）
#   - 使用温度采样（VAL_TEMPERATURE>0 才会真正随机采样）
# 跑完后用 compute_passk.py 聚合 VALIDATION_OUTPUT_DIR/50.jsonl 得到 pass@k。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_EXPERIMENT_NAME=${TRAIN_EXPERIMENT_NAME:?必须提供 TRAIN_EXPERIMENT_NAME}
TRAIN_LOG=${TRAIN_LOG:?必须提供 TRAIN_LOG}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:?必须提供 CHECKPOINT_ROOT}
VALIDATION_LOG=${VALIDATION_LOG:?必须提供 VALIDATION_LOG}
VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR:?必须提供 VALIDATION_OUTPUT_DIR}
TARGET_STEP=${TARGET_STEP:-50}
POLL_SECONDS=${POLL_SECONDS:-60}

# pass@k 采样参数。温度 > 0 才真正采样；=0 时等价 greedy。
VAL_K=${VAL_K:-5}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_TOP_P=${VAL_TOP_P:-0.95}
# SGLang ??????????? vLLM ?????
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
ROLLOUT_SKIP_TOKENIZER_INIT=${ROLLOUT_SKIP_TOKENIZER_INIT:-True}
ROLLOUT_ATTENTION_BACKEND=${ROLLOUT_ATTENTION_BACKEND:-fa3}
export PATH="/root/miniconda3/envs/verl/bin:${PATH}"
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
DO_SAMPLE=False
if awk "BEGIN{exit !(${VAL_TEMPERATURE} > 0)}"; then
    DO_SAMPLE=True
fi

while pgrep -f "trainer.experiment_name=${TRAIN_EXPERIMENT_NAME}" >/dev/null; do
    sleep "${POLL_SECONDS}"
done

STEP_DIR="${CHECKPOINT_ROOT}/global_step_${TARGET_STEP}"
# 训练输出未保存成完整日志文件时，可用 SKIP_PROGRESS_CHECK=True 跳过进度检查，
# 只依赖 checkpoint 目录是否存在（下方仍会校验 ${STEP_DIR}/actor）。
SKIP_PROGRESS_CHECK=${SKIP_PROGRESS_CHECK:-False}
case "${SKIP_PROGRESS_CHECK,,}" in
    true|1|yes) : ;;
    false|0|no)
        if ! grep -aqE 'Training Progress:.*100%' "${TRAIN_LOG}"; then
            echo "训练未正常完成 ${TARGET_STEP} step（进度未达到 100%），不启动验证：${TRAIN_LOG}" >&2
            echo "如果确认训练已跑完，可加 SKIP_PROGRESS_CHECK=True 跳过此检查" >&2
            exit 1
        fi
        ;;
    *) echo "SKIP_PROGRESS_CHECK 必须是 True/False，当前值：${SKIP_PROGRESS_CHECK}" >&2; exit 2;;
esac
if [[ ! -d "${STEP_DIR}/actor" ]]; then
    echo "最终 checkpoint 不存在：${STEP_DIR}" >&2
    exit 1
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
# 纯评测没有梯度回传，单卡 84 GB 默认使用 8 个 Agent Worker 提高吞吐。
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}
export N_RESP_PER_PROMPT=1
export VAL_BEFORE_TRAIN=True
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-200}
export ROLLOUT_DATA_ENABLED=False
export TEST_FREQ=-1
export SAVE_FREQ=-1
export RESUME_MODE=resume_path
export CHECKPOINT_DIR="${CHECKPOINT_ROOT}"
export PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_lora}
export EXPERIMENT_NAME=${VALIDATION_EXPERIMENT_NAME:-${TRAIN_EXPERIMENT_NAME}_passk${VAL_K}_eval}

exec bash "${SCRIPT_DIR}/../train_lora/run_lora.sh" \
    trainer.val_only=True \
    trainer.resume_from_path="${STEP_DIR}" \
    trainer.validation_data_dir="${VALIDATION_OUTPUT_DIR}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_K}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${DO_SAMPLE}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
    actor_rollout_ref.rollout.skip_tokenizer_init="${ROLLOUT_SKIP_TOKENIZER_INIT}" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${ROLLOUT_ATTENTION_BACKEND}" \
    >"${VALIDATION_LOG}" 2>&1
