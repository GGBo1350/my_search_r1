#!/usr/bin/env bash
# 在 AutoDL 现有 verl 环境中把 vLLM 推理后端替换为 SGLang。
# 该脚本不会创建新环境；安装前会把原环境的包版本快照写入数据盘。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_DIR=${ENV_DIR:-/root/miniconda3/envs/verl}
PYTHON=${PYTHON:-${ENV_DIR}/bin/python}
UV_CACHE_DIR=${UV_CACHE_DIR:-/root/autodl-tmp/cache/uv}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-/root/autodl-tmp/env_snapshots}
# AutoDL 系统默认镜像在国内节点通常明显快于官方 PyPI。
PYPI_INDEX=${PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}
PYTORCH_INDEX=${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}
USE_NETWORK_TURBO=${USE_NETWORK_TURBO:-1}

TORCH_VERSION=${TORCH_VERSION:-2.9.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.24.1}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.9.1}
SGLANG_VERSION=${SGLANG_VERSION:-0.5.8}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.8.3}
FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}

export UV_CACHE_DIR
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export PATH="${HOME}/.local/bin:${PATH}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "没有找到现有环境：${PYTHON}" >&2
    exit 1
fi

mkdir -p "${UV_CACHE_DIR}" "${SNAPSHOT_DIR}"

if [[ "${USE_NETWORK_TURBO}" == "1" && -r /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi
# AutoDL 代理偶尔会改写 PyPI/NVIDIA 请求，按平台建议清除代理变量。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

timestamp=$(date +%Y%m%d_%H%M%S)
snapshot="${SNAPSHOT_DIR}/verl_before_sglang_${timestamp}.txt"
{
    "${PYTHON}" --version
    uv pip freeze --python "${PYTHON}"
} >"${snapshot}"
echo "原环境版本快照：${snapshot}"

echo "[1/5] 移除与 Torch 2.9 ABI 不兼容的 vLLM/FlashAttention/xFormers 包"
uv pip uninstall --python "${PYTHON}" vllm flash-attn xformers || true

echo "[2/5] 在原环境升级到 Torch ${TORCH_VERSION} + CUDA 12.8"
uv pip install \
    --python "${PYTHON}" \
    --index-url "${PYTORCH_INDEX}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

echo "[3/5] 安装 SGLang ${SGLANG_VERSION}"
uv pip install \
    --python "${PYTHON}" \
    --default-index "${PYPI_INDEX}" \
    "sglang==${SGLANG_VERSION}" \
    "tensordict==0.10.0"

echo "[4/5] 安装 Torch 2.9 对应的 FlashAttention wheel"
if [[ "${USE_NETWORK_TURBO}" == "1" && -r /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi
uv pip install --python "${PYTHON}" --no-deps "${FLASH_ATTN_WHEEL}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

echo "[5/5] 重新挂载当前仓库并验证依赖"
uv pip install --python "${PYTHON}" --no-deps -e "${REPO_ROOT}"
uv pip check --python "${PYTHON}"

"${PYTHON}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

import flash_attn
import sglang
import torch
import verl

for package in (
    "torch",
    "torchvision",
    "torchaudio",
    "verl",
    "sglang",
    "sgl-kernel",
    "flash-attn",
    "transformers",
    "ray",
    "peft",
):
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT_INSTALLED")

try:
    print(f"vllm: {version('vllm')} (仍存在，请检查安装过程)")
except PackageNotFoundError:
    print("vllm: NOT_INSTALLED")

print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("SGLang/FlashAttention/verl 导入成功")
PY

echo "SGLang 已安装到现有环境：${ENV_DIR}"
echo "无卡模式下 CUDA available=False 属于正常现象。"
