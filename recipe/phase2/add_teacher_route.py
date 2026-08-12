"""Add a top-level bridge/compare teacher route to an existing Parquet file.

The output must not already exist, so this utility never overwrites the source
dataset or a previous migration result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _route_from_extra_info(extra_info: dict[str, Any] | None) -> str:
    info = extra_info or {}
    question_type = str(info.get("question_type") or "").strip().lower()
    strategy = str(info.get("expected_strategy") or "").strip().lower()
    if question_type in {"comparison", "compare"} or strategy in {"parallel", "comparison", "compare"}:
        return "compare"
    if question_type == "bridge" or strategy in {"sequential", "bridge"}:
        return "bridge"
    raise ValueError(f"Cannot infer teacher route from extra_info: {info!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input Parquet does not exist: {args.input}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    table = pq.read_table(args.input)
    if "teacher_route" in table.column_names:
        raise ValueError(f"Input already contains teacher_route: {args.input}")
    if "extra_info" not in table.column_names:
        raise ValueError(f"Input does not contain extra_info: {args.input}")

    routes = [_route_from_extra_info(info) for info in table.column("extra_info").to_pylist()]
    migrated = table.append_column("teacher_route", pa.array(routes, type=pa.string()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(migrated, args.output, compression="zstd")
    print(f"Wrote {len(routes)} rows to {args.output}: {dict(sorted(Counter(routes).items()))}")


if __name__ == "__main__":
    main()
