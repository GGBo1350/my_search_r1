"""Create a small, balanced, non-overwriting OPD smoke-test Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-route", type=int, default=1)
    args = parser.parse_args()

    if args.per_route <= 0:
        raise ValueError("--per-route must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    frame = pd.read_parquet(args.input)
    if "teacher_route" not in frame:
        raise KeyError(f"teacher_route is missing from {args.input}")
    parts = []
    for route in ("bridge", "compare"):
        part = frame.loc[frame["teacher_route"] == route].head(args.per_route)
        if len(part) != args.per_route:
            raise ValueError(f"Expected {args.per_route} {route} rows, found {len(part)}")
        parts.append(part)
    smoke = pd.concat(parts, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    smoke.to_parquet(args.output, index=False)
    print(f"Wrote {len(smoke)} smoke rows to {args.output}: {smoke['teacher_route'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
