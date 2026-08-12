"""Compare aligned Phase 1 greedy rollout dumps with strict answer metrics.

The input JSONL files must contain the same questions in the same order.  The
script deliberately recomputes exact/F1 with ``recipe.core.my_reward`` instead
of trusting runtime ``answer_exact`` fields, which may include an LLM judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.core.my_reward import _answer_metrics

MODEL_ORDER = ("base", "step50", "step100", "answer_only")


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No rollout rows in {path}")
    return rows


def _gold(row: dict[str, Any]) -> Any:
    value = row.get("ground_truth")
    return row.get("gts") if value is None else value


def _route(row: dict[str, Any]) -> str:
    strategy = str(row.get("expected_strategy") or "").strip().lower()
    if strategy == "sequential":
        return "bridge"
    if strategy == "parallel":
        return "compare"
    return strategy or "unlabeled"


def _enrich(row: dict[str, Any]) -> dict[str, Any]:
    f1, exact = _answer_metrics(str(row.get("pred") or ""), _gold(row))
    return {
        **row,
        "strict_f1": f1,
        "strict_exact": exact,
        "route": _route(row),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    return {
        "n": count,
        "exact": sum(bool(row["strict_exact"]) for row in rows) / count,
        "f1_ge_0_5": sum(float(row["strict_f1"]) >= 0.5 for row in rows) / count,
        "mean_f1": sum(float(row["strict_f1"]) for row in rows) / count,
        "strategy": sum(bool(row.get("strategy_correct")) for row in rows) / count,
        "retrieval_recall": sum(float(row.get("retrieval_recall") or 0.0) for row in rows) / count,
        "format_ok": sum(bool(row.get("format_ok")) for row in rows) / count,
        "mean_calls": sum(float(row.get("num_tool_calls") or 0.0) for row in rows) / count,
    }


def _transition(left: list[dict[str, Any]], right: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for left_row, right_row in zip(left, right, strict=True):
        before = int(bool(left_row[field]))
        after = int(bool(right_row[field]))
        counts[f"{before}->{after}"] += 1
    return {key: counts[key] for key in ("0->0", "0->1", "1->0", "1->1")}


def _snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pred": row.get("pred") or "",
        "strict_exact": bool(row["strict_exact"]),
        "strict_f1": float(row["strict_f1"]),
        "strategy_correct": bool(row.get("strategy_correct")),
        "call_group_sizes": row.get("call_group_sizes") or [],
        "queries": row.get("queries") or [],
        "retrieval_recall": float(row.get("retrieval_recall") or 0.0),
        "format_ok": bool(row.get("format_ok")),
    }


def _case(
    index: int,
    aligned: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    base_row = aligned["base"][index]
    return {
        "index": index,
        "question": base_row.get("question") or "",
        "route": base_row["route"],
        "gold": _gold(base_row),
        "models": {model: _snapshot(aligned[model][index]) for model in MODEL_ORDER},
    }


def _candidate_indices(
    aligned: dict[str, list[dict[str, Any]]],
    left: str,
    right: str,
    field: str,
    before: bool,
    after: bool,
    route: str,
) -> list[int]:
    result = []
    for index, (left_row, right_row) in enumerate(zip(aligned[left], aligned[right], strict=True)):
        if route != "all" and left_row["route"] != route:
            continue
        if bool(left_row[field]) == before and bool(right_row[field]) == after:
            result.append(index)
    return result


def _group_sampled(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        question = str(row.get("question") or "")
        grouped.setdefault(question, []).append(row)
    sizes = {len(samples) for samples in grouped.values()}
    if len(sizes) != 1:
        raise ValueError(f"Sample counts differ across questions: {sorted(sizes)}")
    return grouped


def _sampled_summary(groups: dict[str, list[dict[str, Any]]], route: str) -> dict[str, float | int]:
    selected = [samples for samples in groups.values() if route == "all" or samples[0]["route"] == route]
    count = len(selected)
    trajectories = [row for samples in selected for row in samples]
    return {
        "n": count,
        "samples_per_question": len(selected[0]),
        "sample_exact_at_1": sum(bool(samples[0]["strict_exact"]) for samples in selected) / count,
        "pass_at_5_exact": sum(any(bool(row["strict_exact"]) for row in samples) for samples in selected) / count,
        "pass_at_5_f1_ge_0_5": sum(
            any(float(row["strict_f1"]) >= 0.5 for row in samples) for samples in selected
        )
        / count,
        "trajectory_exact": sum(bool(row["strict_exact"]) for row in trajectories) / len(trajectories),
        "trajectory_strategy": sum(bool(row.get("strategy_correct")) for row in trajectories) / len(trajectories),
        "strategy_pass_at_5": sum(
            any(bool(row.get("strategy_correct")) for row in samples) for samples in selected
        )
        / count,
        "strategy_and_exact_pass_at_5": sum(
            any(bool(row.get("strategy_correct")) and bool(row["strict_exact"]) for row in samples)
            for samples in selected
        )
        / count,
    }


def _sampled_case(
    question: str,
    greedy_row: dict[str, Any],
    sampled_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "question": question,
        "route": greedy_row["route"],
        "gold": _gold(greedy_row),
        "greedy": _snapshot(greedy_row),
        "samples": [_snapshot(row) for row in sampled_rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in MODEL_ORDER:
        parser.add_argument(f"--{model.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(f"--{model.replace('_', '-')}-sampled", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON output; stdout is used otherwise.")
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args()

    paths = {
        "base": args.base,
        "step50": args.step50,
        "step100": args.step100,
        "answer_only": args.answer_only,
    }
    aligned = {model: [_enrich(row) for row in _load(path)] for model, path in paths.items()}
    base_questions = [str(row.get("question") or "") for row in aligned["base"]]
    for model in MODEL_ORDER[1:]:
        questions = [str(row.get("question") or "") for row in aligned[model]]
        if questions != base_questions:
            raise ValueError(f"{model} is not question-aligned with base")

    summary = {}
    for model, rows in aligned.items():
        summary[model] = {"all": _summary(rows)}
        for route in ("bridge", "compare"):
            summary[model][route] = _summary([row for row in rows if row["route"] == route])

    transitions = {}
    for left, right in (("base", "step50"), ("step50", "step100"), ("base", "answer_only")):
        key = f"{left}_to_{right}"
        transitions[key] = {}
        for route in ("all", "bridge", "compare"):
            left_rows = aligned[left] if route == "all" else [row for row in aligned[left] if row["route"] == route]
            right_rows = (
                aligned[right] if route == "all" else [row for row in aligned[right] if row["route"] == route]
            )
            transitions[key][route] = {
                "exact": _transition(left_rows, right_rows, "strict_exact"),
                "strategy": _transition(left_rows, right_rows, "strategy_correct"),
            }

    candidate_specs = {
        "base_wrong_step50_right": ("base", "step50", "strict_exact", False, True),
        "base_right_step50_wrong": ("base", "step50", "strict_exact", True, False),
        "step50_wrong_step100_right": ("step50", "step100", "strict_exact", False, True),
        "step50_right_step100_wrong": ("step50", "step100", "strict_exact", True, False),
        "base_right_answer_only_wrong": ("base", "answer_only", "strict_exact", True, False),
        "base_wrong_answer_only_right": ("base", "answer_only", "strict_exact", False, True),
        "base_bad_step50_good_strategy": ("base", "step50", "strategy_correct", False, True),
        "step50_good_step100_bad_strategy": ("step50", "step100", "strategy_correct", True, False),
        "base_good_answer_only_bad_strategy": ("base", "answer_only", "strategy_correct", True, False),
    }
    candidates = {}
    for name, (left, right, field, before, after) in candidate_specs.items():
        candidates[name] = {}
        for route in ("bridge", "compare"):
            indices = _candidate_indices(aligned, left, right, field, before, after, route)
            candidates[name][route] = [
                _case(index, aligned) for index in indices[: args.max_candidates]
            ]

    result = {
        "inputs": {model: str(path) for model, path in paths.items()},
        "summary": summary,
        "transitions": transitions,
        "candidates": candidates,
    }
    sampled_paths = {
        "base": args.base_sampled,
        "step50": args.step50_sampled,
        "step100": args.step100_sampled,
        "answer_only": args.answer_only_sampled,
    }
    provided_sampled = {model for model, path in sampled_paths.items() if path is not None}
    if provided_sampled and provided_sampled != set(MODEL_ORDER):
        raise ValueError(f"Provide sampled dumps for every model, got {sorted(provided_sampled)}")
    if provided_sampled:
        sampled_groups = {
            model: _group_sampled([_enrich(row) for row in _load(path)])
            for model, path in sampled_paths.items()
            if path is not None
        }
        expected_questions = set(base_questions)
        for model, groups in sampled_groups.items():
            if set(groups) != expected_questions:
                raise ValueError(f"{model} sampled questions are not aligned with greedy questions")

        result["sampled_inputs"] = {model: str(path) for model, path in sampled_paths.items()}
        result["sampled_summary"] = {
            model: {
                route: _sampled_summary(groups, route)
                for route in ("all", "bridge", "compare")
            }
            for model, groups in sampled_groups.items()
        }
        sampled_candidates = {}
        for model in MODEL_ORDER:
            greedy_by_question = {str(row.get("question") or ""): row for row in aligned[model]}
            recovery = []
            instability = []
            mixed = []
            for question in base_questions:
                greedy_row = greedy_by_question[question]
                samples = sampled_groups[model][question]
                hits = [bool(row["strict_exact"]) for row in samples]
                case = _sampled_case(question, greedy_row, samples)
                if not bool(greedy_row["strict_exact"]) and any(hits):
                    recovery.append(case)
                if bool(greedy_row["strict_exact"]) and not any(hits):
                    instability.append(case)
                if any(hits) and not all(hits):
                    mixed.append(case)
            sampled_candidates[model] = {
                "greedy_wrong_but_sample_pass5": recovery[: args.max_candidates],
                "greedy_right_but_sample_fail5": instability[: args.max_candidates],
                "mixed_sample_outcomes": mixed[: args.max_candidates],
            }
        result["sampled_candidates"] = sampled_candidates
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
