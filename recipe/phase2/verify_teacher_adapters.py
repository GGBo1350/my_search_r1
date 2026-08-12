"""Validate exported teacher adapters before starting GPU servers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open


def parse_target_modules(value: str) -> set[str]:
    modules = {item.strip().strip("'\"") for item in value.strip("[]").split(",") if item.strip()}
    if not modules:
        raise ValueError("Expected at least one teacher LoRA target module")
    return modules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapters", type=Path, nargs="+")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--target-modules", required=True)
    args = parser.parse_args()

    expected_modules = parse_target_modules(args.target_modules)
    for adapter in args.adapters:
        config_path = adapter / "adapter_config.json"
        weights_path = adapter / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Incomplete teacher adapter: {adapter}")
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        actual_rank = int(config.get("r", 0))
        actual_modules = set(config.get("target_modules") or [])
        if actual_rank != args.rank:
            raise ValueError(f"Teacher adapter rank mismatch for {adapter}: {actual_rank} != {args.rank}")
        if actual_modules != expected_modules:
            raise ValueError(
                f"Teacher adapter target modules mismatch for {adapter}: "
                f"{sorted(actual_modules)} != {sorted(expected_modules)}"
            )

        # Inspect the safetensors header without loading weights.  Every target
        # projection must have both LoRA A and B tensors in the same complete
        # set of transformer layers; adapter_config alone cannot prove this.
        with safe_open(weights_path, framework="pt", device="cpu") as weights:
            weight_keys = list(weights.keys())
        layer_sides = {module: {"A": set(), "B": set()} for module in expected_modules}
        for key in weight_keys:
            layer_match = re.search(r"\.layers\.(\d+)\.", key)
            if layer_match is None:
                continue
            layer = int(layer_match.group(1))
            for module in expected_modules:
                for side in ("A", "B"):
                    if f".{module}.lora_{side}" in key:
                        layer_sides[module][side].add(layer)

        reference_layers = None
        for module, sides in sorted(layer_sides.items()):
            if not sides["A"] or sides["A"] != sides["B"]:
                raise ValueError(
                    f"Incomplete teacher LoRA tensors for {module} in {adapter}: "
                    f"A layers={sorted(sides['A'])}, B layers={sorted(sides['B'])}"
                )
            if reference_layers is None:
                reference_layers = sides["A"]
            elif sides["A"] != reference_layers:
                raise ValueError(
                    f"Teacher LoRA layer coverage differs for {module} in {adapter}: "
                    f"{sorted(sides['A'])} != {sorted(reference_layers)}"
                )
        assert reference_layers is not None
        print(
            f"Verified teacher adapter: {adapter} "
            f"(rank={actual_rank}, target_modules={sorted(actual_modules)}, "
            f"layers={len(reference_layers)}, lora_tensors={sum('lora_' in key for key in weight_keys)})"
        )


if __name__ == "__main__":
    main()
