#!/usr/bin/env python3
"""离线为训练/验证数据生成标准答案的表面变体。

对每一行，把「问题 + 标准答案 + 两条黄金证据文档」交给 DeepSeek，让它只基于
这些材料给出同一答案实体的所有合理表面写法（全名/昵称/别名/带引号形式等），
例如 gold="Steven Williams" 可得到 ["Steven Williams", "Steve Williams",
"\"Dr. Death\" Steve Williams"]。

生成结果写回 reward_model.ground_truth（多答案列表），exact/F1 对任一变体
命中即判对，训练与评测自动一致。每行结果缓存到 JSONL，可断点续跑。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)
PROMPT_VERSION = "answer-variants-v1"
MAX_VARIANTS = 8

SYSTEM_PROMPT = """You are an answer-normalization assistant for QA evaluation.

Given a question, its gold answer, and the gold evidence passages used to derive
it, list EVERY reasonable surface-form variant of the SAME answer entity that a
model could output and still be considered correct: full names, nicknames,
aliases, quoted/ring-name forms, spelled-out forms, common abbreviations, etc.

Rules:
- Only variants of the exact same entity as the gold answer. Never add other
  entities, broader concepts, or explanatory glosses (e.g. "wrestler", "a city").
- Base variants ONLY on the gold answer and the provided evidence passages. Do
  not use outside knowledge or guess.
- Include the gold answer itself first.
- Return JSON only: {"variants": ["...", "..."]}"""


def build_user_payload(row: dict[str, Any]) -> dict[str, Any]:
    extra = row["extra_info"]
    question = str(extra.get("question", "")).strip()
    gold = str(row["reward_model"]["ground_truth"][0]).strip()
    return {
        "question": question,
        "gold_answer": gold,
        "evidence": extract_evidence(extra),
    }


def extract_evidence(extra: dict[str, Any]) -> list[str]:
    """从 extra_info 里取出两条黄金证据（标题 + 对应句子）。"""
    try:
        ctx = extra["tools_kwargs"]["search"]["create_kwargs"]["context"]
        title_to_sentences = dict(zip(ctx.get("title") or [], ctx.get("sentences") or []))
    except (KeyError, TypeError):
        return []
    passages: list[str] = []
    for fact in extra.get("gold_facts") or []:
        title = str(fact.get("title", "")).strip()
        sent_id = fact.get("sent_id", 0)
        sentences = title_to_sentences.get(title) or []
        if not sentences:
            continue
        if isinstance(sent_id, int) and 0 <= sent_id < len(sentences):
            passages.append(f"[{title}] {sentences[sent_id]}")
        else:
            passages.append(f"[{title}] " + " ".join(map(str, sentences[:2])))
        if len(passages) >= 2:
            break
    return passages


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    payload = build_user_payload(row)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Return JSON only:\n" + json.dumps(payload, ensure_ascii=False)},
    ]


def normalize_answer_variant(text: str) -> str:
    lowered = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff ]", " ", lowered)
    return " ".join(cleaned.split())


def sanitize_variants(gold: str, raw: list[Any]) -> list[str]:
    """保留 gold 在最前，按归一化去重，最多 MAX_VARIANTS 个。"""
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        value = str(value).strip()
        key = normalize_answer_variant(value)
        if not value or not key or key in seen:
            return
        seen.add(key)
        variants.append(value[:160])

    add(gold)
    for value in (raw or [])[: MAX_VARIANTS * 3]:
        add(value)
        if len(variants) >= MAX_VARIANTS:
            break
    return variants


async def variant_one(
    client: Any,
    row: dict[str, Any],
    model: str,
    max_tokens: int,
    retries: int,
    timeout: float = 120.0,
) -> list[str]:
    gold = str(row["reward_model"]["ground_truth"][0]).strip()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=build_messages(row),
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("DeepSeek returned empty content")
            payload = json.loads(content)
            raw = payload.get("variants") or []
            if not isinstance(raw, list):
                raise ValueError("response 'variants' is not a list")
            return sanitize_variants(gold, raw)
        except Exception as error:  # SDK exposes several provider-specific exception classes.
            last_error = error
            if attempt + 1 < retries:
                await asyncio.sleep(min(2**attempt, 16) + random.random())
    raise RuntimeError(f"DeepSeek variant generation failed after {retries} attempts: {last_error}") from last_error


def cache_key(split: str, row_id: str, model: str) -> str:
    return f"{split}|{row_id}|{model}"


def load_cache(path: Path, model: str) -> dict[str, list[str]]:
    cached: dict[str, list[str]] = {}
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
                key = cache_key(str(record["split"]), str(record["id"]), model)
                cached[key] = record["variants"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                LOGGER.warning("忽略缓存第 %d 行：%s", line_number, error)
    return cached


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


async def enrich_split(
    client: Any,
    rows: list[dict[str, Any]],
    split: str,
    model: str,
    cache: dict[str, list[str]],
    cache_path: Path,
    max_tokens: int,
    retries: int,
    timeout: float,
    concurrency: int,
) -> list[list[str]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, row: dict[str, Any]) -> tuple[int, list[str]]:
        row_id = str(row["extra_info"]["id"])
        key = cache_key(split, row_id, model)
        if key in cache:
            return index, cache[key]
        async with semaphore:
            variants = await variant_one(client, row, model, max_tokens, retries, timeout)
        append_jsonl(
            cache_path,
            {
                "status": "ok",
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "split": split,
                "id": row_id,
                "variants": variants,
            },
        )
        cache[key] = variants
        return index, variants

    tasks = [one(index, row) for index, row in enumerate(rows)]
    results: list[tuple[int, list[str]]] = []
    for future in asyncio.as_completed(tasks):
        results.append(await future)
    results.sort(key=lambda item: item[0])
    return [variants for _, variants in results]


def apply_variants(row: dict[str, Any], variants: list[str]) -> dict[str, Any]:
    row = dict(row)
    reward_model = dict(row["reward_model"])
    reward_model["ground_truth"] = list(variants)
    row["reward_model"] = reward_model
    extra = dict(row["extra_info"])
    extra["answer_variants"] = list(variants)
    row["extra_info"] = extra
    return row


def read_rows(path: Path) -> tuple[list[dict[str, Any]], pa.Schema]:
    table = pq.read_table(path)
    return table.to_pylist(), table.schema


def write_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Infer the updated nested schema so extra_info.answer_variants is really
    # persisted; forcing the input schema would silently discard that new key.
    table = pa.Table.from_pylist(rows).replace_schema_metadata(schema.metadata)
    pq.write_table(table, path, compression="zstd")


async def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY；请通过环境变量传入，不要把 key 写进脚本")
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise RuntimeError("当前环境缺少 openai SDK，请先安装：pip install openai pyarrow") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / name for name in ("train.parquet", "validation.parquet")]
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError(f"输出已存在于 {args.output_dir}；确认替换时传 --overwrite")

    cache_path = args.cache_file or (args.output_dir / "answer_variants.jsonl")
    cache = load_cache(cache_path, args.model)
    LOGGER.info("缓存命中 %d 条", len(cache))

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    summary: dict[str, Any] = {"model": args.model, "prompt_version": PROMPT_VERSION, "rows": {}}
    for split, source in (("train", args.train_file), ("validation", args.validation_file)):
        rows, schema = read_rows(source)
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        variants = await enrich_split(
            client,
            rows,
            split,
            args.model,
            cache,
            cache_path,
            args.max_tokens,
            args.retries,
            args.timeout,
            args.concurrency,
        )
        updated = [apply_variants(row, variant) for row, variant in zip(rows, variants)]
        output_path = args.output_dir / f"{split}.parquet"
        write_rows(output_path, updated, schema)
        summary["rows"][split] = len(updated)
        LOGGER.info("完成 %s：%d 行 -> %s", split, len(updated), output_path)

    (args.output_dir / "variants_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/hotpotqa_v3_hard_deepseek_2k/train.parquet"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/hotpotqa_v3_hard_deepseek_2k/validation.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hotpotqa_v3_hard_deepseek_2k_variants"))
    parser.add_argument("--cache-file", type=Path, help="变体缓存 JSONL；默认写在 output-dir/answer_variants.jsonl")
    parser.add_argument("--max-rows", type=int, help="每个 split 最多处理多少行（先小批量试跑）")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true", help="替换输出 parquet；保留变体缓存")
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("concurrency/max-tokens/retries/timeout 必须为正")
    if not args.train_file.exists() or not args.validation_file.exists():
        parser.error(f"输入 parquet 不存在：{args.train_file} / {args.validation_file}")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(run(parse_args()))
