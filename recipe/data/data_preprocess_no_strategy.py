#!/usr/bin/env python3
"""Build the no-strategy prompt dataset used by the Exact-only ablation.

The input is an already processed Search-R1 parquet dataset.  This converter
replaces only the system prompt: it removes the explicit Bridge-serial and
Comparison-parallel instructions while preserving tool syntax, evidence
grounding, and concise-answer requirements.  Questions, gold answers, tool
contexts, route labels, and diagnostic metadata remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


NO_STRATEGY_SYSTEM_PROMPT = """Answer the multi-hop factual question using evidence from the local `search` tool.

You are a search-grounded question-answering assistant: every claim in your final
answer MUST be traceable to a retrieved <document>. Never answer from memory.

## Tool protocol

- Begin each assistant turn with <think>...</think> of at most two sentences.
  Decide ONLY what evidence is missing; do NOT restate the question, schema,
  or predict/hallucinate potential search results.
- After that brief plan, choose one action for the turn: call `search`, or
  output the final <answer>. Never mix a tool call and a final answer in one turn.
- Call `search` through the provided native tool interface. Emit the call as a JSON object wrapped in <tool_call>...</tool_call>:
  <tool_call>
  {"name": "search", "arguments": {"query": "a concise entity-focused query"}}
  </tool_call>
- Write a concrete, entity-focused query; never copy schema descriptions or example phrases.
- A tool-use turn must end immediately after the closing </tool_call> so the
  runtime can return the results.
- The runtime supplies <information> in the next message. Never invent or quote a fake tool result.
- Once every step of the reasoning is explicitly supported by retrieved
  evidence, stop searching and output ONLY the short final answer in
  <answer>...</answer>.

## Grounding rules

- NEVER GUESS OR USE INTERNAL MEMORY. Never skip a search step by using internal
  knowledge to infer unmentioned facts. Every factual link used in the answer
  must be verified by a retrieved <document>.
- If a result is not useful, refine the query once. Never repeat the same query.

## Answer format

- If the answer is yes/no, output exactly "yes" or "no" with no explanation.
- Otherwise output the shortest exact answer: a single entity, number, or date.
  Do not restate the question or add commentary.

## Mandatory check before answering

Before outputting <answer>, verify that:
1. You did not use internal memory to skip evidence collection.
2. The exact final answer is explicitly written in a retrieved <document>.
If evidence is missing, execute another search instead of answering.

Now solve the user's question."""


_BANNED_STRATEGY_CUES = (
    "whether the searches are independent or dependent",
    "a parallel batch",
    "For a comparison whose two subjects",
    "When the second subject is unknown",
    "Do not guess the bridge entity",
    "exceed three search calls",
)


def replace_system_prompt(prompt: Any) -> list[dict[str, Any]]:
    """Return a copied chat prompt with the no-strategy system message."""
    messages = json.loads(prompt) if isinstance(prompt, str) else [dict(message) for message in prompt]
    if not messages or messages[0].get("role") != "system":
        raise ValueError("prompt must start with a system message")
    messages[0] = {**messages[0], "content": NO_STRATEGY_SYSTEM_PROMPT}
    return messages


def rewrite_parquet(source: Path, destination: Path) -> int:
    """Rewrite one parquet split while preserving its schema and metadata."""
    table = pq.read_table(source)
    rows = table.to_pylist()
    for row in rows:
        row["prompt"] = replace_system_prompt(row["prompt"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    updated = pa.Table.from_pylist(rows, schema=table.schema)
    updated = updated.replace_schema_metadata(table.schema.metadata)
    pq.write_table(updated, destination, compression="zstd")
    return len(rows)


def validate_output(path: Path) -> None:
    """Fail if an explicit serial/parallel policy cue survived conversion."""
    for index, row in enumerate(pq.read_table(path, columns=["prompt"]).to_pylist()):
        prompt = row["prompt"]
        messages = json.loads(prompt) if isinstance(prompt, str) else prompt
        system_text = str(messages[0]["content"])
        matched = [cue for cue in _BANNED_STRATEGY_CUES if cue.casefold() in system_text.casefold()]
        if matched:
            raise ValueError(f"strategy cue survived in {path} row {index}: {matched}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/hotpotqa_v3_hard_1600/train.parquet"),
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/hotpotqa_v3_hard_1600/validation.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/hotpotqa_v3_no_strategy"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.train_file, args.validation_file):
        if not path.is_file():
            parser.error(f"input parquet does not exist: {path}")
    outputs = [args.output_dir / "train.parquet", args.output_dir / "validation.parquet"]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"output already exists: {existing}; pass --overwrite to replace it")
    return args


def main() -> None:
    args = parse_args()
    for split, source in (("train", args.train_file), ("validation", args.validation_file)):
        output = args.output_dir / f"{split}.parquet"
        count = rewrite_parquet(source, output)
        validate_output(output)
        print(f"{split}: {count} rows -> {output}")


if __name__ == "__main__":
    main()
