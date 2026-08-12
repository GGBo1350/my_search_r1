"""Convert the downloaded HotpotQA distractor dataset to verl parquet files.

The raw dataset is expected to have been saved with ``DatasetDict.save_to_disk``::

    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    ds.save_to_disk("data/hotpot_qa_distractor")

V3 keeps every example's ten candidate paragraphs in ``tools_kwargs``.  The
model can access them only through the local ``search`` tool; gold supporting
facts remain reward-only metadata.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_from_disk

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Answer the multi-hop factual question using evidence from the local `search` tool.

You are a search-grounded question-answering assistant: every claim in your final
answer MUST be traceable to a retrieved <document>. Never answer from memory.

## Tool protocol

- Begin each assistant turn with <think>...</think> of at most two sentences. Decide ONLY what evidence is missing and whether the searches are independent or dependent; do NOT restate the question, schema, or predict/hallucinate potential search results.
- After that brief plan, choose one action for the turn: call `search`, or output the final <answer>. Never mix a tool call and a final answer in one turn.
- Call `search` through the provided native tool interface. Emit each call as a JSON object wrapped in <tool_call>...</tool_call>:
  <tool_call>
  {"name": "search", "arguments": {"query": "a concise entity-focused query"}}
  </tool_call>
- A parallel batch puts every call of the turn inside the same block, one JSON object per line:
  <tool_call>
  {"name": "search", "arguments": {"query": "first subject relevant property"}}
  {"name": "search", "arguments": {"query": "second subject relevant property"}}
  </tool_call>
- Write a concrete, entity-focused query; never copy schema descriptions or example phrases.
- A tool-use turn must end immediately after the closing </tool_call> so the runtime can return the results.
- The runtime supplies <information> in the next message. Never invent or quote a fake tool result.
- Once EVERY step of the reasoning is explicitly supported by retrieved evidence, stop searching and output ONLY the short final answer in <answer>...</answer>.

## Search plan & Grounding Rules

- For a comparison whose two subjects are independently identifiable, issue two different queries in one parallel batch.
- When the second subject is unknown until the first result reveals a bridge entity, issue one query, read the returned information, then issue one new query using that entity.
- ABSOLUTE GROUNDING RULE: NEVER GUESS OR USE INTERNAL MEMORY. Do not guess the bridge entity from memory. Never skip a search step by using your internal knowledge to infer unmentioned facts (e.g., inferring a county from a city, or assuming an entity without searching). EVERY hop in the chain MUST be verified by a retrieved <document>.
- Even if the first result incidentally mentions the final answer, verify the bridge entity with the second focused search.
- If a result is not useful, refine the query once. Never repeat the same query or exceed three search calls in total.

## Answer Format

- If the answer is yes/no, output exactly "yes" or "no" — no explanation.
- Otherwise output the shortest exact answer: a single entity, number, or date. Do not restate the question or add commentary.

## Mandatory Check Before Answering

Before outputting <answer>, verify that:
1. You did NOT use internal memory to skip any search step.
2. The exact final answer is explicitly written in a retrieved <document>.
If evidence is missing, you MUST execute the next search instead of answering.

Now solve the user's question."""


# HotpotQA 中少量 Bridge 问题会把国家答案直接写成国籍/形容词，例如问题中
# 已经出现 "Brazilian"，答案却是 "Brazil"。这里只维护高置信度映射；相比
# 用大模型判断，它是确定、可复现且便于单元测试的。
_COUNTRY_DEMONYMS = {
    "argentina": {"argentine", "argentinian"},
    "australia": {"australian"},
    "austria": {"austrian"},
    "bangladesh": {"bangladeshi"},
    "belgium": {"belgian"},
    "brazil": {"brazilian"},
    "canada": {"canadian"},
    "chile": {"chilean"},
    "china": {"chinese"},
    "colombia": {"colombian"},
    "denmark": {"danish"},
    "egypt": {"egyptian"},
    "england": {"english", "british"},
    "ethiopia": {"ethiopian"},
    "finland": {"finnish"},
    "france": {"french"},
    "germany": {"german"},
    "greece": {"greek"},
    "india": {"indian"},
    "indonesia": {"indonesian"},
    "iran": {"iranian"},
    "iraq": {"iraqi"},
    "ireland": {"irish"},
    "israel": {"israeli"},
    "italy": {"italian"},
    "japan": {"japanese"},
    "kenya": {"kenyan"},
    "malaysia": {"malaysian"},
    "mexico": {"mexican"},
    "netherlands": {"dutch"},
    "new zealand": {"new zealander"},
    "nigeria": {"nigerian"},
    "north korea": {"north korean"},
    "norway": {"norwegian"},
    "pakistan": {"pakistani"},
    "peru": {"peruvian"},
    "philippines": {"filipino", "philippine"},
    "poland": {"polish"},
    "portugal": {"portuguese"},
    "russia": {"russian"},
    "saudi arabia": {"saudi", "saudi arabian"},
    "scotland": {"scottish", "british"},
    "singapore": {"singaporean"},
    "south africa": {"south african"},
    "south korea": {"south korean", "korean"},
    "spain": {"spanish"},
    "sri lanka": {"sri lankan"},
    "sweden": {"swedish"},
    "switzerland": {"swiss"},
    "thailand": {"thai"},
    "turkey": {"turkish"},
    "ukraine": {"ukrainian"},
    "united kingdom": {"british", "english", "scottish", "welsh"},
    "united states": {"american", "u s"},
    "united states of america": {"american", "u s"},
    "uruguay": {"uruguayan"},
    "venezuela": {"venezuelan"},
    "vietnam": {"vietnamese"},
    "wales": {"welsh", "british"},
}
def _leakage_tokens(value: Any) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", normalized)


def _phrase_positions(tokens: list[str], phrase: list[str]) -> list[tuple[int, int]]:
    if not phrase or len(phrase) > len(tokens):
        return []
    width = len(phrase)
    return [(index, index + width) for index in range(len(tokens) - width + 1) if tokens[index : index + width] == phrase]


def _is_explicit_candidate_answer(
    question_tokens: list[str],
    answer_tokens: list[str],
    question_type: str,
) -> bool:
    # Comparison 的答案通常必须原样出现在两个候选实体中，不能把这种正常
    # 题型误判为泄露。
    if question_type == "comparison":
        return True
    positions = _phrase_positions(question_tokens, answer_tokens)
    if not positions:
        return False
    for start, end in positions:
        nearby = question_tokens[max(0, start - 4) : min(len(question_tokens), end + 4)]
        if "or" in nearby or "and" in nearby:
            return True
    return False


def answer_leakage_reason(row: dict[str, Any]) -> str | None:
    """Return a high-confidence reason when the question itself gives the answer."""
    question_tokens = _leakage_tokens(row.get("question", ""))
    answer_tokens = _leakage_tokens(row.get("answer", ""))
    question_type = str(row.get("type", "")).strip().lower()
    if not answer_tokens or answer_tokens in (["yes"], ["no"]):
        return None
    if _is_explicit_candidate_answer(question_tokens, answer_tokens, question_type):
        return None

    if _phrase_positions(question_tokens, answer_tokens):
        return "answer_in_question"

    answer_key = " ".join(answer_tokens)
    for demonym in _COUNTRY_DEMONYMS.get(answer_key, set()):
        if _phrase_positions(question_tokens, demonym.split()):
            return "country_demonym_in_question"
    return None


def build_chat_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def normalize_context(context: dict[str, Any]) -> dict[str, list[Any]]:
    titles = [str(title) for title in _as_list(context.get("title"))]
    sentence_groups = [
        [str(sentence) for sentence in _as_list(sentences)] for sentences in _as_list(context.get("sentences"))
    ]
    if not titles or len(titles) != len(sentence_groups):
        raise ValueError("HotpotQA context must contain equal non-empty title and sentence lists")
    return {"title": titles, "sentences": sentence_groups}


def normalize_supporting_facts(supporting_facts: dict[str, Any]) -> list[dict[str, Any]]:
    titles = _as_list(supporting_facts.get("title"))
    sentence_ids = _as_list(supporting_facts.get("sent_id"))
    if len(titles) != len(sentence_ids):
        raise ValueError("supporting_facts.title and supporting_facts.sent_id must have equal length")
    return [
        {"title": str(title), "sent_id": int(sentence_id)}
        for title, sentence_id in zip(titles, sentence_ids, strict=True)
    ]


def process_row(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    question_type = str(row["type"]).strip().lower()
    if question_type not in {"comparison", "bridge"}:
        raise ValueError(f"Unsupported HotpotQA question type: {question_type!r}")

    context = normalize_context(row["context"])
    gold_facts = normalize_supporting_facts(row["supporting_facts"])
    gold_titles = list(dict.fromkeys(fact["title"] for fact in gold_facts))
    expected_strategy = "parallel" if question_type == "comparison" else "sequential"

    return {
        "data_source": "hotpotqa_distractor_search_r1_v3",
        # Multi-teacher OPD reads its routing key from a top-level DataProto
        # field. Keep this separate from data_source so the reward function
        # continues to use one stable dataset identifier.
        "teacher_route": "compare" if question_type == "comparison" else "bridge",
        "prompt": build_chat_messages(question),
        "ability": "multi_hop_qa",
        "reward_model": {"style": "rule", "ground_truth": [answer]},
        "extra_info": {
            "index": index,
            "id": str(row["id"]),
            "split": split,
            "question": question,
            "question_type": question_type,
            "level": str(row["level"]),
            "expected_strategy": expected_strategy,
            "expected_call_groups": [2] if question_type == "comparison" else [1, 1],
            "gold_titles": gold_titles,
            "gold_facts": gold_facts,
            # Keep only this tool visible even if future versions register more.
            "tool_selection": ["search"],
            "need_tools_kwargs": True,
            "tools_kwargs": {"search": {"create_kwargs": {"context": context}}},
        },
    }


def _iter_processed_rows(
    dataset: Dataset,
    split: str,
    max_samples: int | None,
    bridge_ratio: float | None = None,
    filter_answer_leakage: bool = False,
) -> Iterable[dict[str, Any]]:
    if bridge_ratio is not None:
        if max_samples is None:
            raise ValueError("Stratifying question types requires max_samples")
        if max_samples < 2:
            raise ValueError("Stratifying question types requires at least two samples")
        if not 0 < bridge_ratio < 1:
            raise ValueError("bridge_ratio must be between zero and one")

        bridge_quota = round(max_samples * bridge_ratio)
        bridge_quota = min(max(bridge_quota, 1), max_samples - 1)
        quotas = {
            "bridge": bridge_quota,
            "comparison": max_samples - bridge_quota,
        }
        counts = {question_type: 0 for question_type in quotas}
        skipped_leaks = 0
        for index in range(len(dataset)):
            row = dataset[index]
            if filter_answer_leakage and answer_leakage_reason(row) is not None:
                skipped_leaks += 1
                continue
            question_type = str(row["type"]).strip().lower()
            if question_type not in quotas or counts[question_type] >= quotas[question_type]:
                continue
            yield process_row(row, split=split, index=index)
            counts[question_type] += 1
            if counts == quotas:
                logger.info("Answer-leakage filter skipped %d train rows before quotas were filled", skipped_leaks)
                return

        raise ValueError(
            "Dataset does not contain enough rows to satisfy balanced quotas: "
            f"requested={quotas}, found={counts}"
        )

    selected = 0
    skipped_leaks = 0
    for index in range(len(dataset)):
        if max_samples is not None and selected >= max_samples:
            break
        row = dataset[index]
        if filter_answer_leakage and answer_leakage_reason(row) is not None:
            skipped_leaks += 1
            continue
        yield process_row(row, split=split, index=index)
        selected += 1
    if max_samples is not None and selected < max_samples:
        raise ValueError(f"Dataset contains only {selected} rows after answer-leakage filtering")
    if filter_answer_leakage:
        logger.info("Answer-leakage filter skipped %d rows before selecting %d", skipped_leaks, selected)


def write_parquet(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    batch_size: int = 512,
) -> int:
    """Write rows incrementally so the 90k-example train split stays bounded in RAM."""
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    buffer: list[dict[str, Any]] = []
    count = 0
    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) < batch_size:
                continue
            table = pa.Table.from_pylist(buffer, schema=schema)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(buffer)
            buffer.clear()
            if count % (batch_size * 20) == 0:
                logger.info("Wrote %d rows to %s", count, output_path)

        if buffer:
            table = pa.Table.from_pylist(buffer, schema=schema)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(buffer)
        if writer is None:
            raise ValueError("Cannot write an empty dataset")
    finally:
        if writer is not None:
            writer.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Search-R1 v3 parquet data from HotpotQA distractor")
    parser.add_argument("--input_dir", type=Path, default=Path("data/hotpot_qa_distractor"))
    parser.add_argument("--local_dir", type=Path, default=Path("data/hotpotqa_v3"))
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_validation_samples", type=int)
    parser.add_argument(
        "--bridge_ratio",
        type=float,
        help="Stratify the train subset with this fraction of bridge rows (for example, 0.8)",
    )
    parser.add_argument(
        "--filter_answer_leakage",
        action="store_true",
        help="Filter high-confidence answer leakage from the train split before sampling",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.bridge_ratio is not None and args.max_train_samples is None:
        parser.error("--bridge_ratio requires --max_train_samples")
    if args.bridge_ratio is not None and not 0 < args.bridge_ratio < 1:
        parser.error("--bridge_ratio must be between zero and one")

    loaded = load_from_disk(str(args.input_dir))
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {args.input_dir}, got {type(loaded).__name__}")

    args.local_dir.mkdir(parents=True, exist_ok=True)
    split_specs = (
        ("train", args.max_train_samples),
        ("validation", args.max_validation_samples),
    )
    output_paths = [args.local_dir / f"{split}.parquet" for split, _ in split_specs]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {names}. Pass --overwrite to replace it.")
    for path in existing:
        path.unlink()

    for split, max_samples in split_specs:
        output_path = args.local_dir / f"{split}.parquet"
        count = write_parquet(
            _iter_processed_rows(
                loaded[split],
                split=split,
                max_samples=max_samples,
                bridge_ratio=args.bridge_ratio if split == "train" else None,
                filter_answer_leakage=args.filter_answer_leakage if split == "train" else False,
            ),
            output_path=output_path,
            batch_size=args.batch_size,
        )
        logger.info("Finished %s: %d rows -> %s", split, count, output_path)


if __name__ == "__main__":
    main()
