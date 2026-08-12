"""Validate top-level teacher routing fields in Search-R1 OPD Parquet files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _expected_route(extra_info: dict[str, Any] | None) -> str:
    info = extra_info or {}
    question_type = str(info.get("question_type") or "").strip().lower()
    strategy = str(info.get("expected_strategy") or "").strip().lower()
    if question_type in {"comparison", "compare"} or strategy in {"parallel", "comparison", "compare"}:
        return "compare"
    if question_type == "bridge" or strategy in {"sequential", "bridge"}:
        return "bridge"
    raise ValueError(f"Unknown question type/strategy in extra_info: {info!r}")


def verify(path: Path, require_both: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    schema = pq.read_schema(path)
    required = {"teacher_route", "extra_info"}
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    table = pq.read_table(path, columns=["teacher_route", "extra_info"])
    routes = table.column("teacher_route").to_pylist()
    extra_infos = table.column("extra_info").to_pylist()
    for index, (route, extra_info) in enumerate(zip(routes, extra_infos, strict=True)):
        expected = _expected_route(extra_info)
        if route != expected:
            raise ValueError(f"{path}: row {index} routes to {route!r}, expected {expected!r}")

    counts = Counter(routes)
    unknown = set(counts) - {"bridge", "compare"}
    if unknown:
        raise ValueError(f"{path} contains unknown routes: {sorted(unknown)}")
    if require_both and set(counts) != {"bridge", "compare"}:
        raise ValueError(f"{path} must contain both routes, got {dict(counts)}")
    print(f"OK {path}: {dict(sorted(counts.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, nargs="+")
    parser.add_argument(
        "--allow-single-route",
        action="store_true",
        help="Allow a bridge-only or compare-only file; values and metadata are still checked.",
    )
    args = parser.parse_args()
    for path in args.parquet:
        verify(path, require_both=not args.allow_single_route)


if __name__ == "__main__":
    main()
