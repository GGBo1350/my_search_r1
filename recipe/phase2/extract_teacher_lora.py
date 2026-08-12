"""Extract only the PEFT LoRA adapter from a verl FSDP actor checkpoint.

The source checkpoint is read-only. The output directory must not already
contain files, so repeated runs never overwrite a previous export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def _local_tensor(value):
    """Return the tensor payload for both Tensor and single-rank DTensor values."""
    if hasattr(value, "_local_tensor"):
        return value._local_tensor
    return value


def load_lora_tensors_from_rank_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[tuple[int, ...], tuple[tuple[str, int | None], ...]]]]:
    """Read only local LoRA storages and sharding metadata from one rank.

    FakeTensorMode exposes each storage's byte offset without materializing the
    full checkpoint. This keeps extraction viable when the actor checkpoint is
    much larger than the container memory limit.
    """
    with FakeTensorMode():
        fake_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    selected = {}
    metadata = {}
    with checkpoint_path.open("rb") as checkpoint_file:
        for name, value in fake_state_dict.items():
            if "lora_" not in name:
                continue
            fake_tensor = _local_tensor(value)
            if not isinstance(fake_tensor, torch.Tensor):
                raise TypeError(f"LoRA value is not a tensor: {name} ({type(value).__name__})")
            storage = fake_tensor.untyped_storage()
            offset = getattr(storage, "_checkpoint_offset", None)
            if offset is None:
                raise RuntimeError(
                    "This PyTorch build does not expose checkpoint storage offsets; "
                    "memory-efficient LoRA extraction is unavailable."
                )
            checkpoint_file.seek(offset)
            raw_storage = bytearray(checkpoint_file.read(storage.nbytes()))
            if len(raw_storage) != storage.nbytes():
                raise EOFError(f"Incomplete tensor storage for {name}")
            flat_tensor = torch.frombuffer(raw_storage, dtype=fake_tensor.dtype)
            selected[name] = torch.as_strided(
                flat_tensor,
                size=tuple(fake_tensor.shape),
                stride=tuple(fake_tensor.stride()),
                storage_offset=fake_tensor.storage_offset(),
            ).clone()
            placements = []
            for placement in getattr(value, "placements", ()):
                if placement.is_shard():
                    placements.append(("shard", int(placement.dim)))
                elif placement.is_replicate():
                    placements.append(("replicate", None))
                else:
                    raise NotImplementedError(f"Unsupported LoRA placement for {name}: {placement}")
            metadata[name] = (tuple(value.shape), tuple(placements))

    if not selected:
        raise ValueError(f"Checkpoint contains no LoRA parameters: {checkpoint_path}")
    return selected, metadata


def load_lora_tensors_from_sharded_checkpoint(
    actor_checkpoint: Path, world_size: int
) -> dict[str, torch.Tensor]:
    """Reconstruct LoRA tensors from one-dimensional FSDP rank shards."""
    rank_tensors = []
    reference_metadata = None
    for rank in range(world_size):
        checkpoint_path = actor_checkpoint / f"model_world_size_{world_size}_rank_{rank}.pt"
        tensors, metadata = load_lora_tensors_from_rank_checkpoint(checkpoint_path)
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError(f"LoRA sharding metadata differs at rank {rank}")
        rank_tensors.append(tensors)

    assert reference_metadata is not None
    reference_keys = set(reference_metadata)
    for rank, tensors in enumerate(rank_tensors):
        if set(tensors) != reference_keys:
            raise ValueError(f"LoRA keys differ at rank {rank}")

    merged = {}
    for name, (global_shape, placements) in reference_metadata.items():
        shards = [tensors[name] for tensors in rank_tensors]
        if placements == () or placements == (("replicate", None),):
            tensor = shards[0]
            if any(not torch.equal(tensor, shard) for shard in shards[1:]):
                raise ValueError(f"Replicated LoRA tensor differs across ranks: {name}")
        elif len(placements) == 1 and placements[0][0] == "shard":
            tensor = torch.cat(shards, dim=placements[0][1])
        else:
            raise NotImplementedError(f"Unsupported LoRA placements for {name}: {placements}")
        if tuple(tensor.shape) != global_shape:
            raise ValueError(f"Merged LoRA shape mismatch for {name}: {tuple(tensor.shape)} != {global_shape}")
        merged[name] = tensor
    return merged


def load_lora_tensors_from_single_rank_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Backward-compatible wrapper for existing single-rank callers."""
    tensors, _ = load_lora_tensors_from_rank_checkpoint(checkpoint_path)
    return tensors


def load_lora_tensors_via_device(checkpoint_path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load a checkpoint onto the requested device and retain only CPU LoRA tensors.

    Loading directly onto CUDA avoids the host-RAM peak on GPU machines whose
    container memory limit is smaller than the checkpoint.
    """
    # CPU mmap keeps the multi-gigabyte base weights file-backed and only
    # faults the selected LoRA tensors into memory.  This is important on
    # no-GPU containers whose cgroup memory limit can be much smaller than the
    # host memory reported by ``free``.
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
        mmap=device == "cpu",
    )
    selected = {}
    for name, value in state_dict.items():
        if "lora_" not in name:
            continue
        tensor = _local_tensor(value)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"LoRA value is not a tensor: {name} ({type(value).__name__})")
        selected[name] = tensor.detach().to(device="cpu").clone()
    del state_dict
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not selected:
        raise ValueError(f"Checkpoint contains no LoRA parameters: {checkpoint_path}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="Local Hugging Face base model directory used to build adapter metadata.",
    )
    parser.add_argument(
        "--load-device",
        default="metadata",
        help="Use 'metadata' for storage-range reads or a torch device such as 'cuda' for full device loading.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.actor_checkpoint.is_dir():
        raise FileNotFoundError(f"Actor checkpoint does not exist: {args.actor_checkpoint}")
    if not (args.base_model / "config.json").is_file():
        raise FileNotFoundError(f"Base model config does not exist: {args.base_model / 'config.json'}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(args.actor_checkpoint),
        target_dir=str(args.output),
        hf_model_config_path=str(args.base_model),
    )
    merger = FSDPModelMerger(config)
    world_size = merger._get_world_size()
    if args.load_device == "metadata":
        state_dict = load_lora_tensors_from_sharded_checkpoint(args.actor_checkpoint, world_size)
    else:
        if world_size != 1:
            raise NotImplementedError("Device loading currently supports only single-rank checkpoints")
        checkpoint_path = args.actor_checkpoint / "model_world_size_1_rank_0.pt"
        state_dict = load_lora_tensors_via_device(checkpoint_path, args.load_device)
    adapter_path = merger.save_lora_adapter(state_dict)
    if adapter_path is None:
        raise ValueError(f"Checkpoint contains no LoRA parameters: {args.actor_checkpoint}")
    print(f"Extracted teacher adapter: {adapter_path}")


if __name__ == "__main__":
    main()
