"""Compare greedy teacher-checkpoint rollout dumps with strict answer metrics.

Each positional argument has the form ``LABEL=GLOB``. Example::

    python recipe/phase2/analyze_teacher_checkpoints.py \
        's20=/path/eval_teacher_compare_s20_greedy/*.jsonl' \
        's25=/path/eval_teacher_compare_s25_greedy/*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.core.my_reward import _answer_metrics


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _load(pattern: str) -> list[dict[str, Any]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No rollout dump matches: {pattern}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"Rollout dumps are empty: {pattern}")
    return rows


def _ground_truth(row: dict[str, Any]) -> Any:
    value = row.get("ground_truth")
    if value is None:
        value = row.get("gts")
    return [str(item) for item in value] if isinstance(value, list) else value


def _question_type(row: dict[str, Any]) -> str:
    strategy = str(row.get("expected_strategy") or "").strip().lower()
    if strategy in {"parallel", "compare", "comparison"}:
        return "compare"
    if strategy in {"sequential", "bridge"}:
        return "bridge"
    return strategy or "unlabeled"


def _report_group(label: str, rows: list[dict[str, Any]]) -> None:
    metrics = [_answer_metrics(str(row.get("pred") or ""), _ground_truth(row)) for row in rows]
    f1_values = [f1 for f1, _ in metrics]
    exact_values = [exact for _, exact in metrics]
    strategy_values = [_as_bool(row.get("strategy_correct", False)) for row in rows]
    call_values = [float(row.get("num_tool_calls") or 0) for row in rows]
    count = len(rows)
    print(
        f"  {label:<8} n={count:<4d} "
        f"exact={sum(exact_values) / count:.3f} "
        f"F1>=0.5={sum(f1 >= 0.5 for f1 in f1_values) / count:.3f} "
        f"meanF1={sum(f1_values) / count:.3f} "
        f"strategy={sum(strategy_values) / count:.3f} "
        f"calls={sum(call_values) / count:.2f}"
    )


def report(label: str, pattern: str) -> None:
    rows = _load(pattern)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_question_type(row)].append(row)

    print(f"==== {label} (n={len(rows)}) ====")
    _report_group("all", rows)
    for question_type in ("bridge", "compare"):
        if grouped[question_type]:
            _report_group(question_type, grouped[question_type])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", metavar="LABEL=GLOB")
    args = parser.parse_args()

    for value in args.inputs:
        if "=" not in value:
            parser.error(f"Expected LABEL=GLOB, got {value!r}")
        label, pattern = value.split("=", 1)
        if not label or not pattern:
            parser.error(f"Expected non-empty LABEL=GLOB, got {value!r}")
        report(label, pattern)


if __name__ == "__main__":
    main()
