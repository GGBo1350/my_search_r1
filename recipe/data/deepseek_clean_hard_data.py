#!/usr/bin/env python3
"""使用 DeepSeek 审核 HotpotQA hard 样本，并凑满固定规模的数据集。

默认生成 1600 条训练数据（1200 Bridge、400 Comparison）和 200 条验证数据
（100 Bridge、100 Comparison）。脚本只审核 ``level=hard`` 的候选；达到配额后
立即停止，不会继续清洗完整数据集。每次 API 结果都会追加到 JSONL 缓存，可在
中断后直接续跑。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

try:
    from .data_preprocess import process_row, write_parquet
except ImportError:  # 允许直接执行：python recipe/data/deepseek_clean_hard_data.py
    from data_preprocess import process_row, write_parquet


LOGGER = logging.getLogger(__name__)
PROMPT_VERSION = "hotpotqa-hard-two-evidence-v1"
ALLOWED_PATTERNS = {"bridge_sequential", "comparison_parallel", "other"}

JUDGE_SYSTEM_PROMPT = """You are auditing HotpotQA examples for search-tool training.
Judge only from the supplied question and two gold evidence documents. Do not solve from
your own memory. We want examples that genuinely require consulting both documents.

Definitions:
- direct_answerable: the question alone states the answer or makes it mechanically
  derivable without external factual evidence. A comparison answer appearing as one of
  two candidate entities is NOT leakage by itself.
- single_document_sufficient: the question plus either one gold document is enough to
  determine the exact answer. Do not mark this true merely because the answer string
  occurs there; the document must also justify why it answers the question.
- needs_both_documents: each document supplies a necessary factual hop or comparison
  operand, so removing either document makes the answer unsupported.
- bridge_sequential: the second search subject is discovered from the first document.
- comparison_parallel: both search subjects are already identifiable in the question.
- ideal_search_calls counts focused top-1 document retrievals under a successful
  retriever. Query retries caused by retrieval failure do not count.

Return exactly one JSON object, with no Markdown, using this schema and all fields:
{"direct_answerable": false, "single_document_sufficient": false,
 "needs_both_documents": true, "minimum_evidence_documents": 2,
 "ideal_search_calls": 2, "search_pattern": "bridge_sequential",
 "reason": "short evidence-based reason"}
"""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def gold_documents(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the two gold paragraphs and their annotated supporting sentences."""
    context = row["context"]
    paragraphs = {
        str(title): [str(sentence) for sentence in _as_list(sentences)]
        for title, sentences in zip(
            _as_list(context.get("title")),
            _as_list(context.get("sentences")),
            strict=True,
        )
    }
    facts = row["supporting_facts"]
    support_ids: dict[str, list[int]] = {}
    for title, sentence_id in zip(
        _as_list(facts.get("title")),
        _as_list(facts.get("sent_id")),
        strict=True,
    ):
        support_ids.setdefault(str(title), []).append(int(sentence_id))

    return [
        {
            "title": title,
            "paragraph": " ".join(paragraphs.get(title, [])),
            "supporting_sentences": [
                paragraphs[title][sentence_id]
                for sentence_id in sentence_ids
                if title in paragraphs and 0 <= sentence_id < len(paragraphs[title])
            ],
        }
        for title, sentence_ids in support_ids.items()
    ]


def build_judge_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "question": str(row["question"]),
        "gold_answer_for_audit_only": str(row["answer"]),
        "hotpot_type": str(row["type"]).strip().lower(),
        "hotpot_level": str(row["level"]).strip().lower(),
        "gold_documents": gold_documents(row),
    }
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Audit this example and return JSON only:\n" + json.dumps(payload, ensure_ascii=False),
        },
    ]


def validate_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("DeepSeek output must be a JSON object")
    boolean_fields = ("direct_answerable", "single_document_sufficient", "needs_both_documents")
    for field in boolean_fields:
        if not isinstance(value.get(field), bool):
            raise ValueError(f"{field} must be boolean")
    for field in ("minimum_evidence_documents", "ideal_search_calls"):
        if isinstance(value.get(field), bool) or not isinstance(value.get(field), int):
            raise ValueError(f"{field} must be an integer")
        if not 0 <= value[field] <= 3:
            raise ValueError(f"{field} must be between 0 and 3")
    if value.get("search_pattern") not in ALLOWED_PATTERNS:
        raise ValueError(f"unsupported search_pattern: {value.get('search_pattern')!r}")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return {
        **{field: value[field] for field in boolean_fields},
        "minimum_evidence_documents": value["minimum_evidence_documents"],
        "ideal_search_calls": value["ideal_search_calls"],
        "search_pattern": value["search_pattern"],
        "reason": reason.strip()[:1000],
    }


def accepted(row: dict[str, Any], decision: dict[str, Any]) -> tuple[bool, str]:
    question_type = str(row.get("type", "")).strip().lower()
    level = str(row.get("level", "")).strip().lower()
    expected_pattern = "bridge_sequential" if question_type == "bridge" else "comparison_parallel"
    checks = (
        (level == "hard", "not_hard"),
        (question_type in {"bridge", "comparison"}, "unsupported_type"),
        (len(gold_documents(row)) == 2, "not_two_gold_documents"),
        (not decision["direct_answerable"], "direct_answerable"),
        (not decision["single_document_sufficient"], "single_document_sufficient"),
        (decision["needs_both_documents"], "both_documents_not_required"),
        (decision["minimum_evidence_documents"] == 2, "minimum_documents_not_two"),
        (decision["ideal_search_calls"] == 2, "ideal_calls_not_two"),
        (decision["search_pattern"] == expected_pattern, "strategy_mismatch"),
    )
    for ok, reason in checks:
        if not ok:
            return False, reason
    return True, "accepted"


@dataclass(frozen=True)
class Candidate:
    split: str
    index: int
    row: dict[str, Any]

    @property
    def sample_id(self) -> str:
        return str(self.row["id"])

    @property
    def question_type(self) -> str:
        return str(self.row["type"]).strip().lower()


def cache_key(split: str, sample_id: str, model: str) -> str:
    return f"{PROMPT_VERSION}|{model}|{split}|{sample_id}"


def load_cache(path: Path, model: str) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("status") != "ok" or record.get("model") != model:
                    continue
                if record.get("prompt_version") != PROMPT_VERSION:
                    continue
                decision = validate_decision(record["decision"])
                key = cache_key(str(record["split"]), str(record["id"]), model)
                cached[key] = decision
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                LOGGER.warning("忽略缓存第 %d 行：%s", line_number, error)
    return cached


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


async def judge_one(client: Any, candidate: Candidate, model: str, max_tokens: int, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=build_judge_messages(candidate.row),
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("DeepSeek returned empty content")
            return validate_decision(json.loads(content))
        except Exception as error:  # SDK exposes several provider-specific exception classes.
            last_error = error
            if attempt + 1 < retries:
                await asyncio.sleep(min(2**attempt, 16) + random.random())
    raise RuntimeError(f"DeepSeek audit failed after {retries} attempts: {last_error}") from last_error


def quotas(total: int, bridge_ratio: float) -> dict[str, int]:
    if total < 2:
        raise ValueError("total must be at least 2")
    bridge = round(total * bridge_ratio)
    bridge = min(max(bridge, 1), total - 1)
    return {"bridge": bridge, "comparison": total - bridge}


def hard_candidates(dataset: Dataset, split: str, seed: int) -> list[Candidate]:
    indices = [
        index
        for index in range(len(dataset))
        if str(dataset[index].get("level", "")).strip().lower() == "hard"
        and str(dataset[index].get("type", "")).strip().lower() in {"bridge", "comparison"}
    ]
    random.Random(f"{seed}:{split}").shuffle(indices)
    return [Candidate(split=split, index=index, row=dataset[index]) for index in indices]


async def select_split(
    *,
    client: Any,
    dataset: Dataset,
    split: str,
    split_quotas: dict[str, int],
    model: str,
    cache: dict[str, dict[str, Any]],
    audit_path: Path,
    concurrency: int,
    max_tokens: int,
    retries: int,
    seed: int,
    max_api_calls: int | None,
) -> tuple[list[Candidate], int]:
    selected: dict[str, list[Candidate]] = {"bridge": [], "comparison": []}
    candidates = hard_candidates(dataset, split, seed)
    api_calls = 0
    position = 0

    def needed(question_type: str) -> bool:
        return len(selected[question_type]) < split_quotas[question_type]

    while position < len(candidates) and any(needed(question_type) for question_type in selected):
        batch: list[Candidate] = []
        while position < len(candidates) and len(batch) < concurrency:
            candidate = candidates[position]
            position += 1
            if not needed(candidate.question_type):
                continue
            key = cache_key(split, candidate.sample_id, model)
            decision = cache.get(key)
            if decision is not None:
                keep, _ = accepted(candidate.row, decision)
                if keep and needed(candidate.question_type):
                    selected[candidate.question_type].append(candidate)
                continue
            if max_api_calls is not None and api_calls + len(batch) >= max_api_calls:
                break
            batch.append(candidate)

        if not batch:
            if max_api_calls is not None and api_calls >= max_api_calls:
                break
            continue

        results = await asyncio.gather(
            *(judge_one(client, candidate, model, max_tokens, retries) for candidate in batch),
            return_exceptions=True,
        )
        api_calls += len(batch)
        for candidate, result in zip(batch, results, strict=True):
            base_record = {
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "split": split,
                "index": candidate.index,
                "id": candidate.sample_id,
                "question_type": candidate.question_type,
            }
            if isinstance(result, BaseException):
                append_jsonl(audit_path, {**base_record, "status": "error", "error": str(result)})
                LOGGER.error("审核失败 %s/%s：%s", split, candidate.sample_id, result)
                continue
            keep, reason = accepted(candidate.row, result)
            record = {**base_record, "status": "ok", "accepted": keep, "filter_reason": reason, "decision": result}
            append_jsonl(audit_path, record)
            cache[cache_key(split, candidate.sample_id, model)] = result
            if keep and needed(candidate.question_type):
                selected[candidate.question_type].append(candidate)

        LOGGER.info(
            "%s 进度：Bridge %d/%d，Comparison %d/%d，本次 API 调用 %d",
            split,
            len(selected["bridge"]),
            split_quotas["bridge"],
            len(selected["comparison"]),
            split_quotas["comparison"],
            api_calls,
        )

    counts = {question_type: len(rows) for question_type, rows in selected.items()}
    if counts != split_quotas:
        raise RuntimeError(
            f"{split} 未达到目标配额：目标={split_quotas}，实际={counts}，"
            f"本次 API 调用={api_calls}。可提高 --max-api-calls 或直接重跑以复用缓存。"
        )
    flattened = selected["bridge"] + selected["comparison"]
    random.Random(f"{seed}:{split}:output").shuffle(flattened)
    return flattened, api_calls


def remove_outputs(output_dir: Path) -> None:
    for name in ("train.parquet", "validation.parquet", "selection_summary.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()
    raw_dir = output_dir / "raw_selected"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)


def write_outputs(output_dir: Path, selected: dict[str, list[Candidate]], summary: dict[str, Any]) -> None:
    raw = DatasetDict(
        {
            split: Dataset.from_list([candidate.row for candidate in candidates])
            for split, candidates in selected.items()
        }
    )
    raw.save_to_disk(str(output_dir / "raw_selected"))
    for split, candidates in selected.items():
        rows = (process_row(candidate.row, split=split, index=candidate.index) for candidate in candidates)
        write_parquet(rows, output_dir / f"{split}.parquet")
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def async_main(args: argparse.Namespace) -> None:
    loaded = load_from_disk(str(args.input_dir))
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"期望 DatasetDict，实际为 {type(loaded).__name__}")
    for split in ("train", "validation"):
        if split not in loaded:
            raise KeyError(f"原始数据缺少 {split!r} split")

    targets = {
        "train": {
            "bridge": args.train_bridge,
            "comparison": args.train_comparison,
        },
        "validation": {
            "bridge": args.validation_bridge,
            "comparison": args.validation_comparison,
        },
    }
    candidate_counts = {
        split: {
            question_type: sum(
                str(row.get("level", "")).strip().lower() == "hard"
                and str(row.get("type", "")).strip().lower() == question_type
                for row in loaded[split]
            )
            for question_type in ("bridge", "comparison")
        }
        for split in targets
    }
    LOGGER.info("目标配额：%s", targets)
    LOGGER.info("hard 候选：%s", candidate_counts)
    if args.dry_run:
        print(json.dumps({"targets": targets, "hard_candidates": candidate_counts}, ensure_ascii=False, indent=2))
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY；请通过环境变量传入，不要把 key 写进脚本")
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise RuntimeError("当前环境缺少 openai SDK，请在 verl 环境运行：pip install openai") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in ("train.parquet", "validation.parquet", "raw_selected")]
    if any(path.exists() for path in existing) and not args.overwrite:
        raise FileExistsError(f"输出已存在于 {args.output_dir}；确认替换时传 --overwrite")
    if args.overwrite:
        remove_outputs(args.output_dir)

    audit_path = args.output_dir / "deepseek_audit.jsonl"
    cache = load_cache(audit_path, args.model)
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    selected: dict[str, list[Candidate]] = {}
    api_calls: dict[str, int] = {}
    remaining_api_calls = args.max_api_calls
    for split in ("train", "validation"):
        rows, calls = await select_split(
            client=client,
            dataset=loaded[split],
            split=split,
            split_quotas=targets[split],
            model=args.model,
            cache=cache,
            audit_path=audit_path,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            retries=args.retries,
            seed=args.seed,
            max_api_calls=remaining_api_calls,
        )
        selected[split] = rows
        api_calls[split] = calls
        if remaining_api_calls is not None:
            remaining_api_calls -= calls

    summary = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "seed": args.seed,
        "targets": targets,
        "selected": {
            split: {
                question_type: sum(candidate.question_type == question_type for candidate in candidates)
                for question_type in ("bridge", "comparison")
            }
            for split, candidates in selected.items()
        },
        "api_calls_this_run": api_calls,
        "audit_cache": str(audit_path),
    }
    write_outputs(args.output_dir, selected, summary)
    LOGGER.info("完成：训练 %d 条，验证 %d 条 -> %s", len(selected["train"]), len(selected["validation"]), args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/hotpot_qa_distractor"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hotpotqa_v3_hard_deepseek_1600"))
    parser.add_argument("--train-bridge", type=int, default=1200)
    parser.add_argument("--train-comparison", type=int, default=400)
    parser.add_argument("--validation-bridge", type=int, default=100)
    parser.add_argument("--validation-comparison", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-api-calls", type=int, help="本次运行最多新发起多少次 API 请求；缓存命中不计数")
    parser.add_argument("--dry-run", action="store_true", help="只统计 hard 候选和目标配额，不调用 API")
    parser.add_argument("--overwrite", action="store_true", help="替换最终数据文件；保留 API 审核缓存")
    args = parser.parse_args()
    quota_values = (
        args.train_bridge,
        args.train_comparison,
        args.validation_bridge,
        args.validation_comparison,
    )
    if any(value < 1 for value in quota_values):
        parser.error("all train/validation type quotas must be positive")
    if args.concurrency < 1 or args.max_tokens < 1 or args.retries < 1:
        parser.error("concurrency, max-tokens and retries must be positive")
    if args.max_api_calls is not None and args.max_api_calls < 1:
        parser.error("--max-api-calls must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
