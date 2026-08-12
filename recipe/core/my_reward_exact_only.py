"""Strict Exact Answer-only reward used by the Phase 1 ablation.

The ablation intentionally removes partial-answer, retrieval, strategy, query,
format, duplicate-query, and LLM-judge rewards.  It retains two safeguards from
the main reward implementation:

1. an exact answer is rewarded only after at least one successful search; and
2. calls beyond the fourth receive a 0.05 penalty each.

All diagnostic fields are still returned so the same rollout-analysis scripts
can compare this model with Base and the composite-reward GRPO model.
"""

from __future__ import annotations

import re
from typing import Any

from recipe.core import my_reward as _metrics


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return strict Exact Answer-only reward plus common diagnostics."""
    tokenizer = kwargs.get("tokenizer")
    extra_info = extra_info or {}
    prediction = _metrics.extract_answer(solution_str)
    answer_f1, exact = _metrics._answer_metrics(prediction, ground_truth)

    groups, attempted_calls, _ = _metrics._extract_call_groups(solution_str)
    retrieval_recall, retrieved_titles = _metrics._retrieval_recall(
        solution_str, extra_info.get("gold_titles", [])
    )
    expected_strategy = str(extra_info.get("expected_strategy", "")).strip().lower()
    _, strategy_correct = _metrics._strategy_score(solution_str, groups, expected_strategy)
    query_score, queries = _metrics._query_score(
        solution_str,
        groups,
        question=str(extra_info.get("question", "")),
        gold_titles=extra_info.get("gold_titles", []),
    )
    fmt = _metrics.check_format(solution_str)

    extra_calls = max(0, attempted_calls - _metrics.EXTRA_CALL_PENALTY_THRESHOLD)
    extra_call_penalty = _metrics.EXTRA_CALL_PENALTY * extra_calls
    _, has_duplicate_query = _metrics._duplicate_query_penalty(solution_str, groups)
    think_lengths = _metrics._think_token_lengths(solution_str, tokenizer)
    think_tokens = sum(think_lengths)
    think_total_budget = _metrics.THINK_TOTAL_BUDGET.get(
        expected_strategy, _metrics.THINK_TOTAL_BUDGET["sequential"]
    )
    think_over_budget = think_tokens > think_total_budget or any(
        length > _metrics.THINK_TURN_BUDGET for length in think_lengths
    )

    parsed_calls = sum(len(group["calls"]) for group in groups)
    valid_tool_call_rate = parsed_calls / attempted_calls if attempted_calls else 0.0
    information_blocks = re.findall(
        r"<information(?:\s[^>]*)?>.*?</information>",
        solution_str,
        re.DOTALL | re.IGNORECASE,
    )
    successful_tool_results = sum(
        "<format_error>" not in block.casefold() for block in information_blocks
    )
    tool_execution_success_rate = (
        min(1.0, successful_tool_results / attempted_calls) if attempted_calls else 0.0
    )
    effective_tool_call_rate = valid_tool_call_rate * tool_execution_success_rate

    has_effective_search = parsed_calls > 0 and successful_tool_results > 0
    raw_score = float(exact and has_effective_search) - extra_call_penalty
    score = max(-0.2, min(1.0, raw_score))
    return {
        "score": round(score, 6),
        "question": str(extra_info.get("question", "")),
        "pred": prediction,
        "answer_f1": round(answer_f1, 6),
        "answer_exact": exact,
        "retrieval_recall": round(retrieval_recall, 6),
        "retrieved_titles": retrieved_titles,
        "expected_strategy": expected_strategy or "unlabeled",
        "strategy_correct": strategy_correct,
        "query_score": round(query_score, 6),
        "queries": queries,
        "format_ok": fmt["format_ok"],
        "tool_xml_ok": fmt["tool_xml_ok"],
        "num_tool_calls": attempted_calls,
        "parsed_tool_calls": parsed_calls,
        "valid_tool_call_rate": round(valid_tool_call_rate, 6),
        "tool_execution_success_rate": round(tool_execution_success_rate, 6),
        "effective_tool_call_rate": round(effective_tool_call_rate, 6),
        "think_tokens": think_tokens,
        "think_over_budget_rate": think_over_budget,
        "duplicate_query_rate": has_duplicate_query,
        "call_group_sizes": fmt["call_group_sizes"],
        "extra_call_penalty": extra_call_penalty,
        "raw_score": round(raw_score, 6),
    }
