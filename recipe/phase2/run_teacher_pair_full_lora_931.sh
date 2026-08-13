#!/usr/bin/env bash
# Run the full-LoRA Bridge and Comparison teacher jobs sequentially on one GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIR_RUN_ID=${PAIR_RUN_ID:-$(date +%Y%m%d_%H%M%S)}

echo "PAIR_RUN_ID=${PAIR_RUN_ID}"
echo "Starting Bridge teacher (75 steps; checkpoints 25/50/75)."
RUN_ID="${PAIR_RUN_ID}_bridge" bash "${SCRIPT_DIR}/run_teacher_bridge_full_lora_931.sh" "$@"

echo "Bridge completed. Starting Comparison teacher (25 steps; checkpoint 25)."
RUN_ID="${PAIR_RUN_ID}_compare" bash "${SCRIPT_DIR}/run_teacher_compare_full_lora_931.sh" "$@"

echo "TEACHER_PAIR_DONE at $(date)"
