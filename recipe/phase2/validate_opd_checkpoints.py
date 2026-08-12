"""Validate LoRA tensors in one or more verl actor checkpoints.

The validator reads only LoRA storages through the metadata-aware extractor,
so checking a multi-gigabyte FSDP checkpoint does not materialize the base
model.  This is useful after a save-time FSDP clone warning: evaluation may
proceed only when every expected LoRA tensor is present and finite.
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import torch

from recipe.phase2.extract_teacher_lora import (
    load_lora_tensors_from_sharded_checkpoint,
    load_lora_tensors_via_device,
)


def _parse_modules(value: str) -> set[str]:
    modules = {part.strip().strip("'\"") for part in value.strip("[]").split(",") if part.strip()}
    if not modules:
        raise ValueError("Expected at least one LoRA target module")
    return modules


def _world_size(actor_dir: Path) -> int:
    rank_files = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    if not rank_files:
        raise FileNotFoundError(f"No model rank checkpoint found in {actor_dir}")
    match = re.fullmatch(r"model_world_size_(\d+)_rank_\d+\.pt", rank_files[0].name)
    if match is None:
        raise ValueError(f"Unexpected model checkpoint name: {rank_files[0].name}")
    world_size = int(match.group(1))
    if len(rank_files) != world_size:
        raise ValueError(f"Expected {world_size} rank files in {actor_dir}, found {len(rank_files)}")
    return world_size


def validate(actor_dir: Path, expected_modules: set[str], expected_layers: int) -> None:
    world_size = _world_size(actor_dir)
    if world_size == 1:
        # Single-GPU checkpoints can be memory-mapped directly.  This also
        # avoids a PyTorch 2.8 FakeTensor deserialization regression triggered
        # by metadata-only loading of this checkpoint format.
        checkpoint_path = actor_dir / "model_world_size_1_rank_0.pt"
        tensors = load_lora_tensors_via_device(checkpoint_path, "cpu")
    else:
        tensors = load_lora_tensors_from_sharded_checkpoint(actor_dir, world_size)
    layer_sides = {module: {"A": set(), "B": set()} for module in expected_modules}
    nonfinite: list[str] = []
    empty: list[str] = []
    squared_norm = 0.0

    for name, tensor in tensors.items():
        if tensor.numel() == 0:
            empty.append(name)
        if not torch.isfinite(tensor).all().item():
            nonfinite.append(name)
        squared_norm += tensor.float().square().sum().item()
        layer_match = re.search(r"\.layers\.(\d+)\.", name)
        if layer_match is None:
            continue
        layer = int(layer_match.group(1))
        for module in expected_modules:
            for side in ("A", "B"):
                if f".{module}.lora_{side}" in name:
                    layer_sides[module][side].add(layer)

    if empty:
        raise ValueError(f"Empty LoRA tensors in {actor_dir}: {empty[:5]}")
    if nonfinite:
        raise ValueError(f"Non-finite LoRA tensors in {actor_dir}: {nonfinite[:5]}")

    expected_layer_ids = set(range(expected_layers))
    for module, sides in sorted(layer_sides.items()):
        if sides["A"] != expected_layer_ids or sides["B"] != expected_layer_ids:
            raise ValueError(
                f"Incomplete {module} LoRA coverage in {actor_dir}: "
                f"A={sorted(sides['A'])}, B={sorted(sides['B'])}"
            )

    expected_tensors = expected_layers * len(expected_modules) * 2
    if len(tensors) != expected_tensors:
        raise ValueError(f"LoRA tensor count mismatch in {actor_dir}: {len(tensors)} != {expected_tensors}")
    print(
        f"VALID {actor_dir.parent.name}: tensors={len(tensors)}, layers={expected_layers}, "
        f"modules={','.join(sorted(expected_modules))}, l2_norm={squared_norm**0.5:.6f}"
    )
    del tensors
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor_dirs", type=Path, nargs="+")
    parser.add_argument("--target-modules", required=True)
    parser.add_argument("--expected-layers", type=int, default=36)
    args = parser.parse_args()
    expected_modules = _parse_modules(args.target_modules)
    for actor_dir in args.actor_dirs:
        if not actor_dir.is_dir():
            raise FileNotFoundError(f"Actor checkpoint is missing: {actor_dir}")
        validate(actor_dir, expected_modules, args.expected_layers)


if __name__ == "__main__":
    main()
