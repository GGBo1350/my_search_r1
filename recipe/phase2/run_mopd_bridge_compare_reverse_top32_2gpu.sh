#!/usr/bin/env bash
# Two-teacher, two-GPU OPD with a coarse-grained reverse KL objective.
#
# Routing and teacher checkpoints are identical to the established Top-32
# forward-KL profile:
#   bridge    -> bridge specialist global_step_75
#   comparison -> comparison specialist global_step_25
#
# At every token position the loss is KL(student || teacher) over a shared
# categorical support consisting of the teacher's Top-32 token IDs plus one
# `other` bucket for the remaining vocabulary mass.  This is not the
# single-sample k3 estimator and not a full-vocabulary reverse KL.  It is also
# not OPD-main's `only_stu` objective: that implementation selects student
# Top-K IDs first and requires a local teacher with full logits to score them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DISTILLATION_LOSS_MODE=reverse_kl_topk
export DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}
export DISTILLATION_PROFILE=reverse_top32
export PROJECT_NAME=${PROJECT_NAME:-search_r1_hotpotqa_v3_mopd_reverse_topk}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_mopd_bridge_s75_compare_s25_reverse_top${DISTILLATION_TOPK}_all7_lora_r32_2gpu_$(date +%Y%m%d_%H%M%S)}

echo "DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE}"
echo "DISTILLATION_TOPK=${DISTILLATION_TOPK}"
echo "REVERSE_TOPK_SUPPORT=teacher_topk_plus_other_bucket"

exec bash "${SCRIPT_DIR}/run_mopd_bridge_compare_2gpu_931.sh" "$@"
