"""Summarize aligned Phase 1 checkpoint evaluations with strict metrics.

Each positional input must use ``LABEL=GLOB``.  Answer exact/F1 are
recomputed from ``pred`` and the gold answers, so the report never inherits an
LLM-judge decision from the rollout dump.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
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
        raise ValueError(f"Rollout dump is empty: {pattern}")
    return rows


def _gold(row: dict[str, Any]) -> Any:
    value = row.get("ground_truth")
    return row.get("gts") if value is None else value


def _route(row: dict[str, Any]) -> str:
    strategy = str(row.get("expected_strategy") or "").strip().lower()
    if strategy in {"sequential", "bridge"}:
        return "bridge"
    if strategy in {"parallel", "compare", "comparison"}:
        return "compare"
    return strategy or "unlabeled"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    answer_metrics = [_answer_metrics(str(row.get("pred") or ""), _gold(row)) for row in rows]
    f1_values = [float(f1) for f1, _ in answer_metrics]
    exact_values = [float(bool(exact)) for _, exact in answer_metrics]
    return {
        "n": len(rows),
        "exact": _mean(exact_values),
        "f1_ge_0_5": _mean([float(f1 >= 0.5) for f1 in f1_values]),
        "mean_f1": _mean(f1_values),
        "strategy": _mean([float(_as_bool(row.get("strategy_correct"))) for row in rows]),
        "retrieval_recall": _mean([float(row.get("retrieval_recall") or 0.0) for row in rows]),
        "format_ok": _mean([float(_as_bool(row.get("format_ok"))) for row in rows]),
        "mean_calls": _mean([float(row.get("num_tool_calls") or 0.0) for row in rows]),
    }


def _parse_input(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=GLOB, got {value!r}")
    label, pattern = value.split("=", 1)
    if not label or not pattern:
        raise argparse.ArgumentTypeError(f"Expected non-empty LABEL=GLOB, got {value!r}")
    return label, pattern


def _render_table(summary: dict[str, dict[str, dict[str, float | int]]]) -> str:
    header = (
        f"{'checkpoint':<12} {'group':<8} {'n':>4} {'exact':>7} {'f1>=.5':>7} "
        f"{'meanF1':>7} {'strategy':>8} {'recall':>7} {'format':>7} {'calls':>6}"
    )
    lines = [header, "-" * len(header)]
    for label, groups in summary.items():
        for group in ("all", "bridge", "compare"):
            values = groups.get(group)
            if values is None:
                continue
            lines.append(
                f"{label:<12} {group:<8} {int(values['n']):>4d} "
                f"{float(values['exact']):>7.3f} {float(values['f1_ge_0_5']):>7.3f} "
                f"{float(values['mean_f1']):>7.3f} {float(values['strategy']):>8.3f} "
                f"{float(values['retrieval_recall']):>7.3f} {float(values['format_ok']):>7.3f} "
                f"{float(values['mean_calls']):>6.2f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=_parse_input, metavar="LABEL=GLOB")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--text-output", type=Path)
    parser.add_argument("--expected-count", type=int, default=200)
    args = parser.parse_args()

    loaded: dict[str, list[dict[str, Any]]] = {}
    patterns: dict[str, str] = {}
    for label, pattern in args.inputs:
        if label in loaded:
            parser.error(f"Duplicate label: {label}")
        loaded[label] = _load(pattern)
        patterns[label] = pattern

    reference_questions: list[str] | None = None
    for label, rows in loaded.items():
        if args.expected_count > 0 and len(rows) != args.expected_count:
            raise ValueError(f"{label} contains {len(rows)} rows; expected {args.expected_count}")
        questions = [str(row.get("question") or "") for row in rows]
        if len(set(questions)) != len(questions):
            raise ValueError(f"{label} contains duplicate questions")
        if reference_questions is None:
            reference_questions = questions
        elif questions != reference_questions:
            raise ValueError(f"{label} is not question-aligned with the first checkpoint")

    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for label, rows in loaded.items():
        groups = {
            "all": rows,
            "bridge": [row for row in rows if _route(row) == "bridge"],
            "compare": [row for row in rows if _route(row) == "compare"],
        }
        summary[label] = {name: _summarize(group) for name, group in groups.items() if group}

    result = {"inputs": patterns, "expected_count": args.expected_count, "summary": summary}
    table = _render_table(summary)
    print(table)

    for output in (args.json_output, args.csv_output, args.text_output):
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)

    if args.json_output:
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.text_output:
        args.text_output.write_text(table + "\n", encoding="utf-8")
    if args.csv_output:
        fieldnames = [
            "checkpoint",
            "group",
            "n",
            "exact",
            "f1_ge_0_5",
            "mean_f1",
            "strategy",
            "retrieval_recall",
            "format_ok",
            "mean_calls",
        ]
        with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for label, groups in summary.items():
                for group, metrics in groups.items():
                    writer.writerow({"checkpoint": label, "group": group, **metrics})


if __name__ == "__main__":
    main()
