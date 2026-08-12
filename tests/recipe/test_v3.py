from __future__ import annotations

import asyncio
import bz2
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from recipe.data.build_hotpotqa_db import CorpusDocument, build_database, iter_wikipedia_documents
from datasets import Dataset

from recipe.data.data_preprocess import (
    _iter_processed_rows,
    answer_leakage_reason,
    process_row,
    write_parquet,
)
from recipe.core.my_reward import compute_score
from recipe.core.my_tool_parser import BatchedXMLToolParser
from recipe.core.my_tools import HotpotSearchTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.tool_registry import initialize_tools_from_config
from verl.trainer.ppo.metric_utils import compute_reward_extra_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, should_dump_rollout_data

SEARCH_SCHEMA = OpenAIFunctionToolSchema.model_validate(
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "local search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
)


class StubTokenizer:
    @staticmethod
    def decode(value):
        return value

    @staticmethod
    def encode(value, add_special_tokens=False):
        del add_special_tokens
        return value.split()


def _context() -> dict:
    return {
        "title": ["Alpha City", "Beta River", "Unrelated Topic"],
        "sentences": [
            ["Alpha City was founded in 1901."],
            ["The Beta River crosses Alpha City."],
            ["This paragraph is about botany."],
        ],
    }


def _row(question_type: str = "comparison") -> dict:
    return {
        "id": "sample-id",
        "question": "Were Alpha City and Beta River named in the same year?",
        "answer": "no",
        "type": question_type,
        "level": "medium",
        "supporting_facts": {"title": ["Alpha City", "Beta River"], "sent_id": [0, 0]},
        "context": _context(),
    }


def _extra(expected_strategy: str) -> dict:
    return {
        "question": "Compare Alpha City with Beta River.",
        "gold_titles": ["Alpha City", "Beta River"],
        "expected_strategy": expected_strategy,
        "expected_call_groups": [2] if expected_strategy == "parallel" else [1, 1],
    }


def test_parser_returns_both_calls_from_one_batch_in_text_order():
    parser = BatchedXMLToolParser(StubTokenizer())
    response = (
        "<think>Both subjects are known.</think>"
        "<tool_calls>"
        "<search><query>Alpha City founding</query></search>"
        "<search><query>Beta River naming</query></search>"
        "</tool_calls>"
    )

    _, calls = asyncio.run(parser.extract_tool_calls(response, [SEARCH_SCHEMA]))

    assert parser.stop_strings == ["</tool_calls>"]
    assert [call.name for call in calls] == ["search", "search"]
    assert [json.loads(call.arguments)["query"] for call in calls] == [
        "Alpha City founding",
        "Beta River naming",
    ]
    tool_call_ids = [getattr(call, "tool_call_id", None) for call in calls]
    if any(tool_call_ids):
        # verl v0.9+ carries OpenAI-style call IDs; v0.8 does not expose this
        # field, but still executes both FunctionCall objects independently.
        assert all(tool_call_ids)
        assert len(set(tool_call_ids)) == 2


def test_parser_accepts_qwen_native_parallel_tool_calls():
    parser = BatchedXMLToolParser(StubTokenizer())
    response = (
        '<tool_call>{"name":"search","arguments":{"query":"Arthur Magazine"}}</tool_call>'
        '<tool_call>{"name":"search","arguments":{"query":"First for Women"}}</tool_call>'
    )

    _, calls = asyncio.run(parser.extract_tool_calls(response, [SEARCH_SCHEMA]))

    assert len(calls) == 2
    assert [json.loads(call.arguments)["query"] for call in calls] == [
        "Arthur Magazine",
        "First for Women",
    ]


def test_local_search_ranks_titles_and_releases_state():
    tool = HotpotSearchTool(
        config={"type": "native", "topk": 2, "title_weight": 3},
        tool_schema=SEARCH_SCHEMA,
    )
    instance_id, _ = asyncio.run(tool.create(create_kwargs={"context": _context()}))

    response, reward, metrics = asyncio.run(tool.execute(instance_id, {"query": "Alpha City founding"}))

    assert reward == 0.0
    assert 'title="Alpha City"' in response.text
    assert " rank=" not in response.text
    assert " score=" not in response.text
    assert "<sentence" not in response.text
    assert "Alpha City was founded in 1901.\n</document>" in response.text
    assert metrics["retrieved_titles"][0] == "Alpha City"
    asyncio.run(tool.release(instance_id))
    assert instance_id not in tool._indices


def test_sqlite_search_builds_global_database_and_deduplicates_titles(tmp_path):
    database_path = tmp_path / "hotpotqa.sqlite"
    documents = [
        CorpusDocument(title="Alpha City", sentences=("A short description.",)),
        CorpusDocument(
            title="alpha city",
            sentences=("Alpha City was founded in 1901 and is crossed by the Beta River.",),
        ),
        CorpusDocument(title="Botany", sentences=("Plants convert light into energy.",)),
    ]

    raw_count, unique_count = build_database(documents, database_path, source="unit-test")
    tool = HotpotSearchTool(
        config={
            "type": "native",
            "backend": "sqlite",
            "database_path": str(database_path),
            "topk": 2,
            "title_weight": 3,
        },
        tool_schema=SEARCH_SCHEMA,
    )
    instance_id, _ = asyncio.run(tool.create(create_kwargs={"context": _context()}))
    response, reward, metrics = asyncio.run(tool.execute(instance_id, {"query": "Alpha City founded 1901"}))

    assert raw_count == 3
    assert unique_count == 2
    assert reward == 0.0
    assert metrics["retrieved_titles"][0].casefold() == "alpha city"
    assert "founded in 1901" in response.text
    asyncio.run(tool.release(instance_id))
    assert instance_id not in tool._instances


def test_official_wikipedia_bz2_reader_uses_title_and_text(tmp_path):
    archive_path = tmp_path / "wiki_00.bz2"
    payload = json.dumps(
        {
            "id": "1",
            "title": "Gamma Lake",
            "text": ["Gamma Lake is in the northern region.", "It freezes in winter."],
            "text_with_links": ["<a href=\"x\">Gamma Lake</a> is elsewhere."],
        }
    )
    archive_path.write_bytes(bz2.compress(f"{payload}\n".encode()))

    documents = list(iter_wikipedia_documents(archive_path))

    assert documents == [
        CorpusDocument(
            title="Gamma Lake",
            sentences=("Gamma Lake is in the northern region.", "It freezes in winter."),
        )
    ]


def test_preprocess_hides_strategy_label_from_prompt_and_injects_local_context():
    row = process_row(_row("bridge"), split="train", index=3)
    prompt_text = "\n".join(message["content"] for message in row["prompt"])
    extra = row["extra_info"]

    assert extra["expected_strategy"] == "sequential"
    assert extra["expected_call_groups"] == [1, 1]
    assert extra["tool_selection"] == ["search"]
    assert extra["need_tools_kwargs"] is True
    assert extra["tools_kwargs"]["search"]["create_kwargs"]["context"] == _context()
    assert "sample-id" not in prompt_text
    assert "supporting_facts" not in prompt_text
    assert "provided native tool interface" in prompt_text
    assert "exceed three search calls" in prompt_text
    assert "at most two sentences" in prompt_text
    assert "a concise entity-focused query" not in prompt_text
    assert "exactly two search calls" not in prompt_text
    assert "Do not guess the bridge entity from memory" in prompt_text
    assert row["reward_model"]["ground_truth"] == ["no"]


def test_incremental_parquet_writer_keeps_nested_tool_context(tmp_path):
    rows = [process_row(_row("comparison"), "train", 0), process_row(_row("bridge"), "train", 1)]
    output_path = tmp_path / "v3.parquet"

    count = write_parquet(iter(rows), output_path, batch_size=1)
    loaded = pq.read_table(output_path).to_pylist()

    assert count == 2
    assert loaded[0]["extra_info"]["expected_call_groups"] == [2]
    assert loaded[1]["extra_info"]["expected_call_groups"] == [1, 1]
    assert loaded[0]["extra_info"]["tools_kwargs"]["search"]["create_kwargs"]["context"]["title"][0] == "Alpha City"


def test_stratified_train_subset_uses_configured_bridge_ratio():
    rows = []
    for index, question_type in enumerate(["bridge"] * 8 + ["comparison"] * 3):
        row = _row(question_type)
        row["id"] = f"sample-{index}"
        rows.append(row)

    processed = list(
        _iter_processed_rows(
            Dataset.from_list(rows),
            split="train",
            max_samples=5,
            bridge_ratio=0.8,
        )
    )

    question_types = [row["extra_info"]["question_type"] for row in processed]
    assert question_types.count("bridge") == 4
    assert question_types.count("comparison") == 1
    assert [row["extra_info"]["index"] for row in processed] == [0, 1, 2, 3, 8]


def test_stratified_train_subset_requires_enough_rows():
    with pytest.raises(ValueError, match="balanced quotas"):
        list(
            _iter_processed_rows(
                Dataset.from_list([_row("bridge"), _row("bridge")]),
                split="train",
                max_samples=2,
                bridge_ratio=0.8,
            )
        )


def test_answer_leakage_filter_catches_bridge_leaks_but_keeps_candidates():
    pantanal = _row("bridge")
    pantanal.update(
        question=(
            "The Pantanal was made by the Brazilian manufacturer Troller, "
            "a manufacturer of off-road vehicles in which country?"
        ),
        answer="Brazil",
    )
    radiohead = _row("bridge")
    radiohead.update(
        question="Thom Yorke is best known as the singer of which band, Radiohead?",
        answer="Radiohead",
    )
    explicit_bridge_candidates = _row("bridge")
    explicit_bridge_candidates.update(
        question="Who was older, Anita Lane or Nick Cave?",
        answer="Nick Cave",
    )
    conjunctive_bridge_candidates = _row("bridge")
    conjunctive_bridge_candidates.update(
        question=(
            '"State of Grace" and "I Knew You Were Trouble" were on Red; '
            "which song was written by Swift, Max Martin and Shellback?"
        ),
        answer="I Knew You Were Trouble",
    )
    comparison = _row("comparison")
    comparison.update(
        question="Which magazine started first, Arthur's Magazine or First for Women?",
        answer="Arthur's Magazine",
    )

    assert answer_leakage_reason(pantanal) == "country_demonym_in_question"
    assert answer_leakage_reason(radiohead) == "answer_in_question"
    assert answer_leakage_reason(explicit_bridge_candidates) is None
    assert answer_leakage_reason(conjunctive_bridge_candidates) is None
    assert answer_leakage_reason(comparison) is None


def test_stratified_subset_replenishes_rows_after_answer_leakage_filter():
    rows = []
    for index, question_type in enumerate(["bridge"] * 5 + ["comparison"] * 3):
        row = _row(question_type)
        row["id"] = f"sample-{index}"
        rows.append(row)
    rows[0]["question"] = "A Brazilian manufacturer is based in which country?"
    rows[0]["answer"] = "Brazil"

    processed = list(
        _iter_processed_rows(
            Dataset.from_list(rows),
            split="train",
            max_samples=5,
            bridge_ratio=0.6,
            filter_answer_leakage=True,
        )
    )

    assert len(processed) == 5
    assert [row["extra_info"]["index"] for row in processed] == [1, 2, 3, 5, 6]
    assert sum(row["extra_info"]["question_type"] == "bridge" for row in processed) == 3
    assert sum(row["extra_info"]["question_type"] == "comparison" for row in processed) == 2


def test_comparison_reward_requires_one_two_call_batch():
    parallel = (
        "<think>Independent subjects.</think>"
        "<tool_calls>"
        "<search><query>Alpha City history</query></search>"
        "<search><query>Beta River history</query></search>"
        "</tool_calls>"
        '<information><document title="Alpha City">A</document></information>'
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )
    split = (
        "<think>Independent subjects.</think>"
        "<tool_calls><search><query>Alpha City history</query></search></tool_calls>"
        '<information><document title="Alpha City">A</document></information>'
        "<tool_calls><search><query>Beta River history</query></search></tool_calls>"
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )

    good = compute_score(parallel, ["no"], _extra("parallel"))
    bad = compute_score(split, ["no"], _extra("parallel"))

    assert good["score"] == 1.0
    assert good["strategy_correct"] is True
    assert good["retrieval_recall"] == 1.0
    assert good["valid_tool_call_rate"] == 1.0
    assert good["tool_execution_success_rate"] == 1.0
    assert good["effective_tool_call_rate"] == 1.0
    assert good["raw_score"] == 1.0
    assert bad["strategy_correct"] is False
    assert bad["score"] == 0.85


def test_answer_reward_requires_at_least_one_effective_search():
    direct_answer = "<think>I remember the answer.</think><answer>no</answer>"
    failed_search = (
        "<think>Search first.</think>"
        "<tool_calls><search><query>Alpha City history</query></search></tool_calls>"
        "<information><format_error>search failed</format_error></information>"
        "<answer>no</answer>"
    )
    failed_search_wrong_answer = failed_search.replace("<answer>no</answer>", "<answer>yes</answer>")

    direct_reward = compute_score(direct_answer, ["no"], _extra("parallel"))
    failed_reward = compute_score(failed_search, ["no"], _extra("parallel"))
    failed_wrong_reward = compute_score(failed_search_wrong_answer, ["no"], _extra("parallel"))

    # 答案指标用于评估，仍应如实记录；但二者都没有成功搜索，因此不能给
    # Answer F1/Exact 奖励。失败搜索仍可获得少量查询/策略行为分，但答对与
    # 答错的得分必须相同。
    assert direct_reward["answer_f1"] == 1.0
    assert direct_reward["answer_exact"] is True
    assert direct_reward["effective_tool_call_rate"] == 0.0
    assert direct_reward["raw_score"] == 0.0
    assert direct_reward["score"] == 0.0
    assert failed_reward["answer_f1"] == 1.0
    assert failed_reward["answer_exact"] is True
    assert failed_reward["tool_execution_success_rate"] == 0.0
    assert failed_wrong_reward["answer_f1"] == 0.0
    assert failed_wrong_reward["answer_exact"] is False
    assert failed_reward["raw_score"] == failed_wrong_reward["raw_score"]
    assert failed_reward["score"] == failed_wrong_reward["score"]


def test_reward_exposes_only_compact_scalar_metrics():
    response = (
        "<think>Independent subjects.</think>"
        "<tool_calls>"
        "<search><query>Alpha City history</query></search>"
        "<search><query>Beta River history</query></search>"
        "</tool_calls>"
        '<information><document title="Alpha City">A</document></information>'
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )

    reward = compute_score(response, ["no"], _extra("parallel"))
    scalar_keys = {
        key for key, value in reward.items() if isinstance(value, (bool, int, float, np.number))
    }

    assert scalar_keys == {
        "score",
        "answer_f1",
        "answer_exact",
        "retrieval_recall",
        "strategy_correct",
        "query_score",
        "format_ok",
        "tool_xml_ok",
        "num_tool_calls",
        "parsed_tool_calls",
        "valid_tool_call_rate",
        "tool_execution_success_rate",
        "effective_tool_call_rate",
        "think_tokens",
        "think_over_budget_rate",
        "duplicate_query_rate",
        "extra_call_penalty",
        "raw_score",
    }


def test_reward_penalizes_overlong_think_with_strategy_specific_budget():
    short_parallel = (
        "<think>brief plan</think>"
        "<tool_calls>"
        "<search><query>Alpha City history</query></search>"
        "<search><query>Beta River history</query></search>"
        "</tool_calls>"
        '<information><document title="Alpha City">A</document></information>'
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )
    long_parallel = short_parallel.replace("brief plan", " ".join(["reason"] * 801))

    short_reward = compute_score(short_parallel, ["no"], _extra("parallel"), tokenizer=StubTokenizer())
    long_reward = compute_score(long_parallel, ["no"], _extra("parallel"), tokenizer=StubTokenizer())

    assert short_reward["think_tokens"] == 2
    assert short_reward["think_over_budget_rate"] is False
    assert long_reward["think_tokens"] == 801
    assert long_reward["think_over_budget_rate"] is True
    assert long_reward["score"] < short_reward["score"]


def test_reward_allows_bridge_think_up_to_1000_tokens_across_turns():
    sequential = (
        f"<think>{' '.join(['first'] * 500)}</think>"
        "<tool_calls><search><query>Alpha City history</query></search></tool_calls>"
        '<information><document title="Alpha City">Beta River</document></information>'
        f"<think>{' '.join(['second'] * 500)}</think>"
        "<tool_calls><search><query>Beta River history</query></search></tool_calls>"
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )

    reward = compute_score(sequential, ["no"], _extra("sequential"), tokenizer=StubTokenizer())

    assert reward["think_tokens"] == 1000
    assert reward["think_over_budget_rate"] is False
    assert reward["score"] == 1.0


def test_reward_penalizes_normalized_duplicate_successful_queries():
    repeated = (
        "<think>Find the bridge.</think>"
        "<tool_calls><search><query>Alpha City history!</query></search></tool_calls>"
        '<information><document title="Alpha City">Beta River</document></information>'
        "<think>Try again.</think>"
        "<tool_calls><search><query> alpha city HISTORY </query></search></tool_calls>"
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )

    reward = compute_score(repeated, ["no"], _extra("sequential"), tokenizer=StubTokenizer())

    assert reward["duplicate_query_rate"] is True
    assert reward["query_score"] == 1.0
    assert reward["score"] == 0.9


def test_failed_tool_call_does_not_count_as_duplicate_query():
    retry = (
        "<think>Search.</think>"
        "<tool_calls><search><query>Alpha City history</query></search></tool_calls>"
        "<information><format_error>temporary failure</format_error></information>"
        "<think>Retry.</think>"
        "<tool_calls><search><query>Alpha City history</query></search></tool_calls>"
        '<information><document title="Alpha City">A</document></information>'
        "<answer>no</answer>"
    )

    reward = compute_score(retry, ["no"], _extra("sequential"), tokenizer=StubTokenizer())

    assert reward["duplicate_query_rate"] is False


def test_bridge_reward_requires_information_between_single_call_batches():
    sequential = (
        "<think>Find the bridge.</think>"
        "<tool_calls><search><query>Alpha City history</query></search></tool_calls>"
        '<information><document title="Alpha City">It is crossed by Beta River.</document></information>'
        "<think>Use the bridge entity.</think>"
        "<tool_calls><search><query>Beta River history</query></search></tool_calls>"
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )
    batched = (
        "<think>Guess both.</think>"
        "<tool_calls>"
        "<search><query>Alpha City history</query></search>"
        "<search><query>Beta River history</query></search>"
        "</tool_calls>"
        '<information><document title="Alpha City">A</document></information>'
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )

    good = compute_score(sequential, ["no"], _extra("sequential"))
    bad = compute_score(batched, ["no"], _extra("sequential"))

    assert good["score"] == 1.0
    assert good["strategy_correct"] is True
    assert bad["strategy_correct"] is False
    assert bad["score"] == 0.85


def test_reward_groups_qwen_native_calls_by_tool_result_boundary():
    comparison = (
        '<tool_call>{"name":"search","arguments":{"query":"Alpha City history"}}</tool_call>'
        '<tool_call>{"name":"search","arguments":{"query":"Beta River history"}}</tool_call>'
        '<information><document title="Alpha City">A</document></information>'
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )
    bridge = (
        '<tool_call>{"name":"search","arguments":{"query":"Alpha City history"}}</tool_call>'
        '<information><document title="Alpha City">Beta River</document></information>'
        '<tool_call>{"name":"search","arguments":{"query":"Beta River history"}}</tool_call>'
        '<information><document title="Beta River">B</document></information>'
        "<answer>no</answer>"
    )

    comparison_score = compute_score(comparison, ["no"], _extra("parallel"))
    bridge_score = compute_score(bridge, ["no"], _extra("sequential"))

    assert comparison_score["call_group_sizes"] == [2]
    assert comparison_score["strategy_correct"] is True
    assert comparison_score["format_ok"] is True
    assert comparison_score["score"] == 1.0
    assert bridge_score["call_group_sizes"] == [1, 1]
    assert bridge_score["strategy_correct"] is True
    assert bridge_score["score"] == 1.0


def test_v3_tool_config_registers_only_local_search():
    config_path = Path(__file__).parents[2] / "recipe" / "v3" / "tool_config.yaml"
    tools = initialize_tools_from_config(config_path)

    assert len(tools) == 1
    assert tools[0].name == "search"
    assert isinstance(tools[0], HotpotSearchTool)
    assert tools[0].topk == 1


def test_reward_extra_metrics_only_aggregate_scalar_curves():
    metrics = compute_reward_extra_metrics(
        {
            "retrieval_recall": np.array([0.0, 0.5, 1.0], dtype=object),
            "strategy_correct": np.array([False, True, True], dtype=object),
            "queries": np.array([["alpha"], ["beta"], ["gamma"]], dtype=object),
            "pred": np.array(["a", "b", "c"], dtype=object),
        }
    )

    assert metrics["reward_extra/retrieval_recall/mean"] == 0.5
    assert metrics["reward_extra/strategy_correct/mean"] == 2 / 3
    assert "reward_extra/queries/mean" not in metrics
    assert "reward_extra/pred/mean" not in metrics


def test_rollout_dump_toggle_frequency_and_review_fields(tmp_path):
    assert should_dump_rollout_data(5, str(tmp_path), 5) is True
    assert should_dump_rollout_data(6, str(tmp_path), 5) is False
    assert should_dump_rollout_data(5, None, 5) is False
    with pytest.raises(ValueError, match="positive integer"):
        should_dump_rollout_data(5, str(tmp_path), 0)

    RayPPOTrainer._write_generations(
        inputs=["system prompt and question"],
        outputs=["<answer>no</answer>"],
        gts=[["no"]],
        scores=[0.75],
        reward_extra_infos_dict={
            "question": ["Were Alpha City and Beta River named in the same year?"],
            "pred": ["no"],
            "queries": [["Alpha City", "Beta River"]],
            "retrieved_titles": [["Alpha City", "Beta River"]],
        },
        dump_path=str(tmp_path),
        global_steps=5,
    )

    record = json.loads((tmp_path / "5.jsonl").read_text(encoding="utf-8"))
    assert record["question"].startswith("Were Alpha City")
    assert record["input"] == "system prompt and question"
    assert record["ground_truth"] == ["no"]
    assert record["gts"] == ["no"]
    assert record["output"] == "<answer>no</answer>"
    assert record["score"] == 0.75
    assert record["queries"] == ["Alpha City", "Beta River"]
