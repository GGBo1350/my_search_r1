#!/usr/bin/env python3
"""下载 HotpotQA distractor，并保存为可离线加载的 DatasetDict。"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hotpot_qa_distractor"),
        help="DatasetDict.save_to_disk 的输出目录",
    )
    parser.add_argument("--force", action="store_true", help="目标存在时仍重新下载")
    return parser.parse_args()


def print_summary(dataset: DatasetDict, prefix: str) -> None:
    sizes = ", ".join(f"{name}={len(split):,}" for name, split in dataset.items())
    print(f"{prefix}: {sizes}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists() and not args.force:
        try:
            existing = load_from_disk(str(output_dir))
        except (FileNotFoundError, ValueError):
            pass
        else:
            if not isinstance(existing, DatasetDict):
                raise TypeError(f"目标不是 DatasetDict：{output_dir}")
            print_summary(existing, "数据已经存在")
            print(f"路径：{output_dir}")
            return 0

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
    if not isinstance(dataset, DatasetDict):
        raise TypeError("hotpotqa/hotpot_qa distractor 未返回 DatasetDict")
    print_summary(dataset, "下载完成")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print_summary(load_from_disk(str(output_dir)), "落盘校验通过")
    print(f"路径：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
