"""Rule reward for HotpotQA search quality and call-batch topology."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import string
import unicodedata
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

ANSWER_F1_WEIGHT = 0.35
ANSWER_EXACT_WEIGHT = 0.15
RETRIEVAL_WEIGHT = 0.30
STRATEGY_WEIGHT = 0.15
QUERY_WEIGHT = 0.0
FORMAT_WEIGHT = 0.05
EXTRA_CALL_PENALTY = 0.05
EXTRA_CALL_PENALTY_THRESHOLD = 4
# ---- 语义等价 LLM 判定（0 < F1 < 1 的部分匹配）----
ANSWER_LLM_JUDGE = os.environ.get("ANSWER_LLM_JUDGE", "1") != "0"
_LLM_JUDGE_CLIENT = None

SYSTEM_LLM_JUDGE = """You are an answer-equivalence judge for QA evaluation.

Given the retrieved documents, a predicted answer, and the gold answer(s), decide
whether the predicted answer is semantically equivalent to the gold answer: the
same entity or fact, possibly phrased differently (e.g. "Steve Williams" vs
"Steven Williams", "Boston" vs "Greater Boston Area").

Rules:
- The gold answers are the ONLY reference. The documents are context.
- Return match=true if the predicted answer expresses the same entity/fact as
  ANY gold answer; otherwise match=false.
- Do NOT be lenient about different entities or wrong facts.
Return JSON only: {"match": true} or {"match": false}"""
# Think 长度仅作为观测指标（think_tokens / think_over_budget_rate），不参与奖励：
# 困难多跳题需要足够思考，惩罚会让模型过早停止推理。
THINK_TURN_BUDGET = 512
THINK_TOTAL_BUDGET = {"parallel": 800, "sequential": 1000}
DUPLICATE_QUERY_PENALTY = 0.10
DUPLICATE_QUERY_MAX_PENALTY = 0.20
# Keep in sync with my_tools.py max_query_chars (default 70): a query longer
# than this is rejected by the tool as a format error, so the reward must not
# credit it either.
MAX_QUERY_CHARS = 70

_OUTER_GROUP_RE = re.compile(r"<tool_calls>\s*(.*?)\s*</tool_calls>", re.DOTALL | re.IGNORECASE)
_SEARCH_CALL_RE = re.compile(
    r"<search>\s*<query>\s*(.*?)\s*</query>\s*</search>",
    re.DOTALL | re.IGNORECASE,
)
_NATIVE_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", re.IGNORECASE)
_PLACEHOLDERS = {"query", "search query", "your query", "your search query", "test", "xxx"}


def _decode_json_objects(body: str) -> tuple[list[dict[str, Any]], bool]:
    """Decode one or more consecutive JSON objects from a tool-call block body.

    A parallel batch packs every call of a turn into a single
    ``<tool_call>...</tool_call>`` block with one JSON object per line.  The
    block is valid only when every object decodes and only whitespace remains.
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    length = len(body)
    while index < length:
        while index < length and body[index] in " \t\r\n":
            index += 1
        if index >= length:
            return objects, True
        try:
            obj, end = decoder.raw_decode(body, index)
        except json.JSONDecodeError:
            return objects, False
        if not isinstance(obj, dict):
            return objects, False
        objects.append(obj)
        index = end
    return objects, True


def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else ""


def normalize_answer(text: str) -> str:
    """HotpotQA-style English answer normalization."""
    lowered = str(text).lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _canonical_answer(raw_answer: Any) -> list[str]:
    """Return alternative exact-match targets for gloss-style gold answers.

    HotpotQA gold answers occasionally append an explanatory annotation, e.g.
    ``"Peshwa" (Prime Minister)``.  Answering the core entity (``Peshwa``)
    OR the gloss (``Prime Minister``) is treated as fully correct, so each
    meaningful part becomes an independent exact-match candidate.
    """
    text = str(raw_answer)
    quoted = re.findall(r'"([^"]*)"', text)
    parenthesized = re.findall(r"\(([^)]*)\)", text)
    stripped = re.sub(r"\s*\([^)]*\)\s*", " ", text).replace('"', " ")
    candidates = [" ".join(stripped.split())]
    candidates.extend(" ".join(part.split()) for part in quoted + parenthesized)
    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduplicated.append(candidate)
    return deduplicated


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _answer_metrics(prediction: str, ground_truth: Any) -> tuple[float, bool]:
    prediction_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    exact = False
    for raw_answer in _as_list(ground_truth):
        answer_tokens = normalize_answer(str(raw_answer)).split()
        exact = exact or bool(prediction_tokens) and prediction_tokens == answer_tokens

        # 带解释性括号/引号的标准答案（如 "\"Peshwa\" (Prime Minister)"）
        # 会生成多个候选（核心实体 Peshwa、注释 Prime Minister、整体）。模型
        # 答中任一候选即视为完全正确，给满 F1 与 exact。
        if bool(prediction_tokens):
            for canonical in _canonical_answer(raw_answer):
                canonical_tokens = normalize_answer(canonical).split()
                if canonical_tokens and prediction_tokens == canonical_tokens:
                    exact = True
                    best_f1 = max(best_f1, 1.0)
                    break

        if not prediction_tokens or not answer_tokens:
            continue
        prediction_special = prediction_tokens[0] if len(prediction_tokens) == 1 else ""
        answer_special = answer_tokens[0] if len(answer_tokens) == 1 else ""
        special_answers = {"yes", "no", "noanswer"}
        if (
            prediction_special in special_answers or answer_special in special_answers
        ) and prediction_tokens != answer_tokens:
            continue
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        overlap = sum(common.values())
        if overlap:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(answer_tokens)
            best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return best_f1, exact


def _think_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of <think>...</think> reasoning blocks.

    Tool-call syntax inside reasoning is planning text, not an action, so the
    call counters must ignore it.
    """
    return [
        (match.start(), match.end())
        for match in re.finditer(r"<think>.*?</think>", text, re.DOTALL | re.IGNORECASE)
    ]


def _tool_response_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of tool responses that must never count as calls.

    The tool returns every result and error inside <information>...</information>,
    and the local infer script additionally wraps each response in
    <tool_response>. Tags inside these blocks are retrieval evidence or format
    guidance, not assistant output, so the call counters must ignore them.
    """
    ranges: list[tuple[int, int]] = []
    pattern = re.compile(
        r"<tool_response>.*?</tool_response>|<information(?:\s[^>]*)?>.*?</information>",
        re.DOTALL | re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        ranges.append((match.start(), match.end()))
    return ranges


def _extract_call_groups(text: str) -> tuple[list[dict[str, Any]], int, bool]:
    groups: list[dict[str, Any]] = []
    parsed_calls = 0
    group_bodies_valid = True
    legacy_ranges: list[tuple[int, int]] = []
    excluded_ranges = _tool_response_ranges(text) + _think_ranges(text)

    def overlaps_excluded(start: int, end: int) -> bool:
        return any(
            start < excluded_end and end > excluded_start
            for excluded_start, excluded_end in excluded_ranges
        )

    for match in _OUTER_GROUP_RE.finditer(text):
        if overlaps_excluded(match.start(), match.end()):
            continue
        legacy_ranges.append((match.start(), match.end()))
        calls = []
        for call_match in _SEARCH_CALL_RE.finditer(match.group(1)):
            calls.append(
                {
                    "query": html.unescape(call_match.group(1).strip()),
                    "start": match.start(1) + call_match.start(),
                    "end": match.start(1) + call_match.end(),
                }
            )
        residual = _SEARCH_CALL_RE.sub("", match.group(1)).strip()
        if residual or not calls:
            group_bodies_valid = False
        groups.append({"calls": calls, "start": match.start(), "end": match.end()})
        parsed_calls += len(calls)

    native_matches = [
        match
        for match in _NATIVE_CALL_RE.finditer(text)
        if not overlaps_excluded(match.start(), match.end())
    ]
    native_group: dict[str, Any] | None = None
    attempted_native_objects = 0
    for match in native_matches:
        if any(start <= match.start() < end for start, end in legacy_ranges):
            continue
        payloads, body_valid = _decode_json_objects(match.group(1))
        block_calls: list[dict[str, Any]] = []
        if body_valid:
            for payload in payloads:
                try:
                    arguments = payload.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    query = arguments.get("query") if isinstance(arguments, dict) else None
                except (AttributeError, TypeError, ValueError):
                    body_valid = False
                    break
                if payload.get("name") != "search" or not isinstance(query, str) or not query.strip():
                    body_valid = False
                    break
                block_calls.append({"query": html.unescape(query.strip())})
        if not body_valid or not block_calls:
            group_bodies_valid = False
            attempted_native_objects += max(1, len(payloads))
            continue
        attempted_native_objects += len(payloads)
        if native_group is None or "<information" in text[native_group["end"] : match.start()].lower():
            native_group = {"calls": [], "start": match.start(), "end": match.end()}
            groups.append(native_group)
        for call in block_calls:
            native_group["calls"].append(
                {
                    **call,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        native_group["end"] = match.end()
        parsed_calls += len(block_calls)

    groups.sort(key=lambda group: group["start"])
    attempted_legacy_calls = len(
        [
            match
            for match in re.finditer(r"<search(?:\s|>)", text, re.IGNORECASE)
            if not overlaps_excluded(match.start(), match.end())
        ]
    )
    attempted_native_calls = len(
        [
            match
            for match in re.finditer(r"<tool_call(?:\s|>)", text, re.IGNORECASE)
            if not overlaps_excluded(match.start(), match.end())
        ]
    )
    attempted_calls = attempted_legacy_calls + attempted_native_objects
    attempted_outer_groups = len(
        [
            match
            for match in re.finditer(r"<tool_calls(?:\s|>)", text, re.IGNORECASE)
            if not overlaps_excluded(match.start(), match.end())
        ]
    )
    complete = (
        group_bodies_valid
        and parsed_calls == attempted_calls
        and len(legacy_ranges) == attempted_outer_groups
        and len(native_matches) == attempted_native_calls
        and attempted_calls > 0
    )
    return groups, attempted_calls, complete


def _retrieved_titles(text: str) -> list[str]:
    return [
        html.unescape(match.group(2)).strip()
        for match in re.finditer(r"<document\b[^>]*\btitle=(['\"])(.*?)\1", text, re.IGNORECASE)
    ]


def _normalize_title(title: str) -> str:
    return " ".join(html.unescape(str(title)).casefold().split())


def _retrieval_recall(text: str, gold_titles: Any) -> tuple[float, list[str]]:
    gold = {_normalize_title(title) for title in _as_list(gold_titles) if str(title).strip()}
    retrieved_raw = _retrieved_titles(text)
    retrieved = {_normalize_title(title) for title in retrieved_raw}
    recall = len(gold & retrieved) / len(gold) if gold else 0.0
    return recall, retrieved_raw


def _strategy_score(text: str, groups: list[dict[str, Any]], expected_strategy: str) -> tuple[float, bool]:
    group_sizes = [len(group["calls"]) for group in groups]
    if expected_strategy == "parallel":
        valid = group_sizes == [2]
    elif expected_strategy == "sequential":
        has_intermediate_result = False
        if len(groups) >= 2:
            between = text[groups[0]["end"] : groups[1]["start"]]
            has_intermediate_result = bool(
                re.search(r"<information(?:\s[^>]*)?>.*?</information>", between, re.DOTALL | re.IGNORECASE)
            )
        valid = group_sizes == [1, 1] and has_intermediate_result
    else:
        valid = False
    return (1.0 if valid else 0.0), valid


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(html.unescape(str(text)).casefold()))


def _normalize_query(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(str(query))).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character for character in normalized
    )
    return " ".join(without_punctuation.split())


def _successful_queries(text: str, groups: list[dict[str, Any]]) -> list[str]:
    """Return parsed queries whose corresponding local-tool result is not an error."""
    successful: list[str] = []
    for index, group in enumerate(groups):
        region_end = groups[index + 1]["start"] if index + 1 < len(groups) else len(text)
        region = text[group["end"] : region_end]
        information_blocks = re.findall(
            r"<information(?:\s[^>]*)?>.*?</information>", region, re.DOTALL | re.IGNORECASE
        )
        for call, information in zip(group["calls"], information_blocks):
            if "<format_error>" not in information.casefold():
                successful.append(call["query"])
    return successful


def _duplicate_query_penalty(text: str, groups: list[dict[str, Any]]) -> tuple[float, bool]:
    normalized = [query for raw in _successful_queries(text, groups) if (query := _normalize_query(raw))]
    duplicate_count = len(normalized) - len(set(normalized))
    penalty = min(DUPLICATE_QUERY_MAX_PENALTY, DUPLICATE_QUERY_PENALTY * duplicate_count)
    return penalty, duplicate_count > 0


def _think_token_lengths(text: str, tokenizer: Any | None) -> list[int]:
    thoughts = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    lengths: list[int] = []
    for thought in thoughts:
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                lengths.append(len(tokenizer.encode(thought, add_special_tokens=False)))
                continue
            except (AttributeError, TypeError, ValueError):
                pass
        # Unit tests and standalone callers may not provide the actor tokenizer.
        # Production training passes it from NaiveRewardManager.
        lengths.append(len(_TOKEN_RE.findall(html.unescape(thought).casefold())))
    return lengths



def _query_score(
    text: str,
    groups: list[dict[str, Any]],
    question: str,
    gold_titles: Any,
) -> tuple[float, list[str]]:
    calls = [call for group in groups for call in group["calls"]]
    queries = [call["query"] for call in calls]
    if not calls:
        return 0.0, queries

    reference_tokens = _tokens(question)
    for title in _as_list(gold_titles):
        reference_tokens.update(_tokens(str(title)))

    validity: list[float] = []
    previous_information = ""
    for call in calls:
        query = " ".join(call["query"].split())
        query_tokens = _tokens(query)
        basic_valid = 1 < len(query) <= MAX_QUERY_CHARS and query.casefold() not in _PLACEHOLDERS and bool(query_tokens)
        relevant = bool(query_tokens & (reference_tokens | _tokens(previous_information)))
        validity.append(float(basic_valid and relevant))
        following = text[call["end"] :]
        information = re.search(
            r"<information(?:\s[^>]*)?>.*?</information>",
            following,
            re.DOTALL | re.IGNORECASE,
        )
        if information:
            previous_information += " " + information.group(0)

    return sum(validity) / len(validity), queries


def check_format(text: str) -> dict[str, Any]:
    groups, attempted_calls, tool_xml_ok = _extract_call_groups(text)
    think_ok = bool(re.search(r"<think>.*?</think>", text, re.DOTALL | re.IGNORECASE))
    answer_matches = list(re.finditer(r"<answer>.*?</answer>", text, re.DOTALL | re.IGNORECASE))
    answer_ok = bool(answer_matches)
    calls_before_answer = not answer_matches or all(group["end"] < answer_matches[-1].start() for group in groups)
    # Non-reasoning instruct models such as Qwen2.5 may correctly call tools
    # without emitting a <think> block. Keep it as a metric, not a requirement.
    format_ok = answer_ok and tool_xml_ok and calls_before_answer
    return {
        "format_ok": format_ok,
        "think_ok": think_ok,
        "answer_ok": answer_ok,
        "tool_xml_ok": tool_xml_ok,
        "calls_before_answer": calls_before_answer,
        "num_tool_calls": attempted_calls,
        "call_group_sizes": [len(group["calls"]) for group in groups],
    }


def _llm_semantic_match(
    solution_str: str,
    prediction: str,
    ground_truth: Any,
    question: str = "",
) -> tuple[bool, bool]:
    """用大模型判断预测答案与标准答案是否语义等价。

    仅用于 0 < F1 < 1 的部分匹配。调用失败会重试一次，仍失败返回
    (False, True)，由调用方回退到规则强匹配。返回 (match, error_flag)。
    """
    global _LLM_JUDGE_CLIENT
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return False, True
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai SDK 未安装，跳过 LLM 语义判定")
        return False, True

    documents = [
        html.unescape(re.sub(r"<[^>]+>", "", match).strip())
        for match in re.findall(
            r"<document\b[^>]*>(.*?)</document>", solution_str, re.DOTALL | re.IGNORECASE
        )
    ][:5]
    payload = {
        "question": str(question or ""),
        "retrieved_documents": documents,
        "predicted_answer": prediction,
        "gold_answers": [str(value) for value in _as_list(ground_truth)],
    }
    messages = [
        {"role": "system", "content": SYSTEM_LLM_JUDGE},
        {"role": "user", "content": "Return JSON only:\n" + json.dumps(payload, ensure_ascii=False)},
    ]
    if _LLM_JUDGE_CLIENT is None:
        _LLM_JUDGE_CLIENT = OpenAI(
            api_key=api_key,
            base_url=(
                os.environ.get("ANSWER_LLM_BASE_URL")
                or os.environ.get("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com"
            ),
            timeout=float(os.environ.get("ANSWER_LLM_TIMEOUT", "30")),
        )
    last_error: Exception | None = None
    for _ in range(2):
        try:
            response = _LLM_JUDGE_CLIENT.chat.completions.create(
                model=(
                    os.environ.get("ANSWER_LLM_MODEL")
                    or os.environ.get("DEEPSEEK_MODEL")
                    or "deepseek-v4-flash"
                ),
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=256,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("empty content")
            return bool(json.loads(content).get("match")), False
        except Exception as error:
            last_error = error
    logger.warning("LLM 答案语义判定失败（已重试一次），回退强匹配：%s", last_error)
    return False, True


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score answer correctness and the search behavior that produced it."""
    tokenizer = kwargs.get("tokenizer")
    extra_info = extra_info or {}
    prediction = extract_answer(solution_str)
    answer_f1, exact = _answer_metrics(prediction, ground_truth)
    if ANSWER_LLM_JUDGE and prediction and answer_f1 > 0.0 and not exact:
        judged, judge_error = _llm_semantic_match(
            solution_str, prediction, ground_truth, question=str(extra_info.get("question", ""))
        )
        if not judge_error:
            answer_f1 = 1.0 if judged else 0.0
            exact = bool(judged)

    groups, attempted_calls, _ = _extract_call_groups(solution_str)
    retrieval_recall, retrieved_titles = _retrieval_recall(solution_str, extra_info.get("gold_titles", []))
    expected_strategy = str(extra_info.get("expected_strategy", "")).strip().lower()
    strategy_score, strategy_correct = _strategy_score(solution_str, groups, expected_strategy)
    query_score, queries = _query_score(
        solution_str,
        groups,
        question=str(extra_info.get("question", "")),
        gold_titles=extra_info.get("gold_titles", []),
    )
    fmt = check_format(solution_str)

    expected_calls_raw = extra_info.get("expected_call_groups", [2])
    expected_calls = sum(int(value) for value in _as_list(expected_calls_raw))
    extra_calls = max(0, attempted_calls - EXTRA_CALL_PENALTY_THRESHOLD)
    extra_call_penalty = EXTRA_CALL_PENALTY * extra_calls
    duplicate_penalty, has_duplicate_query = _duplicate_query_penalty(solution_str, groups)
    think_lengths = _think_token_lengths(solution_str, tokenizer)
    think_tokens = sum(think_lengths)
    think_total_budget = THINK_TOTAL_BUDGET.get(expected_strategy, THINK_TOTAL_BUDGET["sequential"])
    think_over_budget = think_tokens > think_total_budget or any(
        length > THINK_TURN_BUDGET for length in think_lengths
    )

    parsed_calls = sum(len(group["calls"]) for group in groups)
    valid_tool_call_rate = parsed_calls / attempted_calls if attempted_calls else 0.0
    information_blocks = re.findall(
        r"<information(?:\s[^>]*)?>.*?</information>", solution_str, re.DOTALL | re.IGNORECASE
    )
    successful_tool_results = sum("<format_error>" not in block.casefold() for block in information_blocks)
    tool_execution_success_rate = (
        min(1.0, successful_tool_results / attempted_calls) if attempted_calls else 0.0
    )
    effective_tool_call_rate = valid_tool_call_rate * tool_execution_success_rate

    # V3 的答案必须建立在真实搜索结果之上。原先即使完全跳过工具，模型也能凭
    # 参数记忆命中答案并获得最多 0.45 的答案奖励，长跑后会逐渐学会直接作答。
    # 只要至少有一次可解析且成功执行的搜索，答案奖励就恢复；原始 F1/Exact
    # 指标仍照常返回，便于验证时区分“答对了”和“按要求完成了搜索”。
    has_effective_search = parsed_calls > 0 and successful_tool_results > 0
    answer_reward_gate = float(has_effective_search)
    answer_f1_reward = ANSWER_F1_WEIGHT * answer_f1 * answer_reward_gate
    answer_exact_reward = ANSWER_EXACT_WEIGHT * float(exact) * answer_reward_gate
    retrieval_reward = RETRIEVAL_WEIGHT * retrieval_recall
    strategy_reward = STRATEGY_WEIGHT * strategy_score
    query_reward = QUERY_WEIGHT * query_score
    format_reward = FORMAT_WEIGHT * float(fmt["format_ok"])
    raw_score = (
        answer_f1_reward
        + answer_exact_reward
        + retrieval_reward
        + strategy_reward
        + query_reward
        + format_reward
        - extra_call_penalty
        - duplicate_penalty
    )
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
