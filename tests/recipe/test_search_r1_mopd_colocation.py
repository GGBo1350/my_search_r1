import importlib.util
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.teacher_loop import teacher_model as teacher_model_module
from verl.experimental.teacher_loop.teacher_model import _prepare_teacher_model_kwargs
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.utils import Role

_EXTRACT_SCRIPT = Path(__file__).parents[2] / "recipe" / "phase2" / "extract_teacher_lora.py"
_EXTRACT_SPEC = importlib.util.spec_from_file_location("extract_teacher_lora", _EXTRACT_SCRIPT)
_EXTRACT_MODULE = importlib.util.module_from_spec(_EXTRACT_SPEC)
assert _EXTRACT_SPEC.loader is not None
_EXTRACT_SPEC.loader.exec_module(_EXTRACT_MODULE)


class _FakeTeacherManager:
    def __init__(self, distillation_config, teacher_model_config, resource_pool):
        self.resource_pool = resource_pool
        self.server_addresses = [teacher_model_config.key]
        self.server_handles = [teacher_model_config.key]
        self.load_balancer_handle = teacher_model_config.key


def _manager(*, pool_world_size=1, max_colocate_count=3, teacher_world_sizes=(1, 1)):
    teachers = OrderedDict(
        (key, SimpleNamespace(key=key, world_size=world_size))
        for key, world_size in zip(("bridge", "compare"), teacher_world_sizes, strict=True)
    )
    manager = teacher_model_module.MultiTeacherModelManager.__new__(
        teacher_model_module.MultiTeacherModelManager
    )
    manager.distillation_config = SimpleNamespace(teacher_models=teachers, colocate_with_actor=True)
    manager.resource_pool = SimpleNamespace(
        world_size=pool_world_size,
        max_colocate_count=max_colocate_count,
    )
    manager.teacher_model_managers = {}
    manager.server_addresses = {}
    manager.server_handles = {}
    manager.load_balancer_handle = {}
    return manager


def test_colocated_teachers_reuse_actor_pool(monkeypatch):
    monkeypatch.setattr(teacher_model_module, "TeacherModelManager", _FakeTeacherManager)
    manager = _manager()

    manager._initialize_teacher_model_managers()

    assert set(manager.teacher_model_managers) == {"bridge", "compare"}
    assert all(item.resource_pool is manager.resource_pool for item in manager.teacher_model_managers.values())


def test_static_sglang_teacher_lora_is_registered():
    teacher = SimpleNamespace(
        key="bridge",
        model_path="/models/qwen3-4b",
        lora_adapter_path="/models/bridge-s75/lora_adapter",
        lora_rank=32,
        lora_target_modules=["qkv_proj", "o_proj"],
        inference=SimpleNamespace(name="sglang", engine_kwargs={}),
    )

    kwargs = _prepare_teacher_model_kwargs(teacher)

    assert kwargs["path"] == "/models/qwen3-4b"
    assert kwargs["lora_rank"] == 32
    assert teacher.inference.engine_kwargs["sglang"]["lora_paths"] == [
        "verl_actor_lora_name=/models/bridge-s75/lora_adapter"
    ]


def test_memory_efficient_lora_extraction_reads_exact_tensor_values(tmp_path):
    checkpoint_path = tmp_path / "model_world_size_1_rank_0.pt"
    lora_a = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    lora_b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    torch.save(
        {
            "base_model.model.layer.weight": torch.full((64, 64), 7.0),
            "base_model.model.layer.lora_A.default.weight": lora_a,
            "base_model.model.layer.lora_B.default.weight": lora_b,
        },
        checkpoint_path,
    )

    selected = _EXTRACT_MODULE.load_lora_tensors_from_single_rank_checkpoint(checkpoint_path)

    assert set(selected) == {
        "base_model.model.layer.lora_A.default.weight",
        "base_model.model.layer.lora_B.default.weight",
    }
    torch.testing.assert_close(selected["base_model.model.layer.lora_A.default.weight"], lora_a)
    torch.testing.assert_close(selected["base_model.model.layer.lora_B.default.weight"], lora_b)


def test_device_lora_extraction_reads_exact_tensor_values(tmp_path):
    checkpoint_path = tmp_path / "model_world_size_1_rank_0.pt"
    lora = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    torch.save(
        {
            "base_model.model.layer.weight": torch.full((64, 64), 7.0),
            "base_model.model.layer.lora_A.default.weight": lora,
        },
        checkpoint_path,
    )

    selected = _EXTRACT_MODULE.load_lora_tensors_via_device(checkpoint_path, "cpu")

    assert set(selected) == {"base_model.model.layer.lora_A.default.weight"}
    torch.testing.assert_close(selected["base_model.model.layer.lora_A.default.weight"], lora)


@pytest.mark.parametrize(
    ("pool_world_size", "max_colocate_count", "teacher_world_sizes", "message"),
    [
        (1, 2, (1, 1), "max_colocate_count"),
        (1, 3, (1, 2), "same world size"),
    ],
)
def test_colocation_rejects_incompatible_pool(
    monkeypatch, pool_world_size, max_colocate_count, teacher_world_sizes, message
):
    monkeypatch.setattr(teacher_model_module, "TeacherModelManager", _FakeTeacherManager)
    manager = _manager(
        pool_world_size=pool_world_size,
        max_colocate_count=max_colocate_count,
        teacher_world_sizes=teacher_world_sizes,
    )

    with pytest.raises(ValueError, match=message):
        manager._initialize_teacher_model_managers()


def _resource_config(colocate_with_actor: bool):
    return OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
            "reward": {
                "reward_model": {
                    "enable": False,
                    "enable_resource_pool": False,
                    "n_gpus_per_node": 0,
                    "nnodes": 0,
                }
            },
            "distillation": {
                "enabled": True,
                "n_gpus_per_node": 2,
                "nnodes": 1,
                "colocate_with_actor": colocate_with_actor,
            },
        }
    )


@pytest.mark.parametrize("colocate", [True, False])
def test_task_runner_teacher_resource_mapping(colocate):
    config = _resource_config(colocate)
    runner = TaskRunner()
    runner.add_teacher_model_resource_pool(config)
    resource_manager = runner.init_resource_pool_mgr(config)

    expected_pool = "global_pool" if colocate else "teacher_pool"
    assert runner.mapping[Role.TeacherModel] == expected_pool
    assert ("teacher_pool" in resource_manager.resource_pool_spec) is (not colocate)
