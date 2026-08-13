from pathlib import Path


_ROOT = Path(__file__).parents[2]
_PHASE2 = _ROOT / "recipe" / "phase2"


def _script(name: str) -> str:
    return (_PHASE2 / name).read_text(encoding="utf-8")


def test_931_mopd_launcher_is_fixed_to_forward_top32_and_full_lora():
    script = _script("run_mopd_bridge_compare_2gpu_931.sh")

    assert "DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-forward_kl_topk}" in script
    assert "DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}" in script
    assert "DISTILLATION_PROFILE=${DISTILLATION_PROFILE:-forward_top32}" in script
    assert "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" in script
    assert "prepare_full_lora_teacher_pair_931.sh" in script
    assert "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}" in script
    assert "USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}" in script
    assert "RESUME_MODE=disable" in script


def test_reverse_top32_wrapper_explicitly_selects_reverse_profile():
    script = _script("run_mopd_bridge_compare_reverse_top32_2gpu.sh")

    assert "DISTILLATION_LOSS_MODE=reverse_kl_topk" in script
    assert "DISTILLATION_PROFILE=reverse_top32" in script


def test_serial_k3_launcher_runs_bridge_then_compare_with_state_only_handoff():
    script = _script("run_serial_bridge_then_compare_sample_k3_1gpu_931.sh")

    bridge_stage = script.index("STAGE_NAME=bridge")
    compare_stage = script.index("STAGE_NAME=compare")
    assert bridge_stage < compare_stage
    assert "BRIDGE_STEP=\"${BRIDGE_CHECKPOINT_DIR}/global_step_75\"" in script
    assert "HANDOFF_STEP" in script
    assert 'ln -s "${BRIDGE_STEP}/actor" "${HANDOFF_STEP}/actor"' in script
    assert '"${HANDOFF_STEP}/data.pt"' in script
    assert "TRAIN_BATCH_SIZE=16 TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=100" in script
    assert "TRAIN_BATCH_SIZE=16 TOTAL_EPOCHS=4 TOTAL_TRAINING_STEPS=100" in script
    assert "SAVE_FREQ=25 MAX_ACTOR_CKPT_TO_KEEP=3" in script
    assert "for step in 25 50 75" in script
    assert "SAVE_FREQ=100 MAX_ACTOR_CKPT_TO_KEEP=1" in script
    assert script.count('STUDENT_MODEL="${BASE_MODEL}"') == 2
    assert '"${COMPARE_CHECKPOINT_DIR}/global_step_100/actor"' in script


def test_single_teacher_serial_stage_is_reward_free_sample_k3():
    script = _script("run_single_teacher_sample_k3_1gpu_931.sh")

    assert "distillation.distillation_loss.loss_mode=k3" in script
    assert "distillation.distillation_loss.topk=null" in script
    assert "distillation.distillation_loss.use_task_rewards=False" in script
    assert "distillation.distillation_loss.use_policy_gradient=False" in script
    assert "NGPUS_PER_NODE=1 NNODES=1 ROLLOUT_TP=1" in script
    assert "distillation.n_gpus_per_node=1" in script
    assert "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1" in script
    assert "trainer.del_local_ckpt_after_load=False" in script
