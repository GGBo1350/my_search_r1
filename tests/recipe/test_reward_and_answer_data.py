from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from recipe.core.my_reward_exact_only import compute_score as compute_exact_only_score
from recipe.data.data_preprocess_no_strategy import (
    NO_STRATEGY_SYSTEM_PROMPT,
    rewrite_parquet,
    validate_output,
)
from recipe.data.enrich_answer_variants import apply_variants, write_rows


def _successful_search(answer: str) -> str:
    return (
        '<tool_call>{"name":"search","arguments":{"query":"Alpha City history"}}</tool_call>'
        '<information><document title="Alpha City">Evidence.</document></information>'
        f"<answer>{answer}</answer>"
    )


def test_exact_only_reward_has_no_partial_credit_and_requires_search():
    exact = compute_exact_only_score(_successful_search("Alpha City"), ["Alpha City"])
    partial = compute_exact_only_score(_successful_search("Alpha"), ["Alpha City"])
    direct = compute_exact_only_score("<answer>Alpha City</answer>", ["Alpha City"])

    assert exact["answer_exact"] is True and exact["score"] == 1.0
    assert 0.0 < partial["answer_f1"] < 1.0 and partial["score"] == 0.0
    assert direct["answer_exact"] is True and direct["score"] == 0.0


def test_no_strategy_converter_changes_only_prompt(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    row = {
        "prompt": [
            {
                "role": "system",
                "content": "For a comparison whose two subjects are known, use a parallel batch.",
            },
            {"role": "user", "content": "Question?"},
        ],
        "reward_model": {"style": "rule", "ground_truth": ["answer"]},
        "extra_info": {"id": "row-1", "expected_strategy": "parallel"},
    }
    pq.write_table(pa.Table.from_pylist([row]), source)

    assert rewrite_parquet(source, output) == 1
    validate_output(output)
    converted = pq.read_table(output).to_pylist()[0]

    assert converted["prompt"][0]["content"] == NO_STRATEGY_SYSTEM_PROMPT
    assert converted["prompt"][1] == row["prompt"][1]
    assert converted["reward_model"] == row["reward_model"]
    assert converted["extra_info"] == row["extra_info"]


def test_answer_variants_are_persisted_in_nested_extra_info(tmp_path):
    source_row = {
        "index": 7,
        "reward_model": {"style": "rule", "ground_truth": ["Steven Williams"]},
        "extra_info": {"id": "row-1", "question": "Who?"},
    }
    source_schema = pa.Table.from_pylist([source_row]).schema
    variants = ["Steven Williams", "Steve Williams"]
    updated = apply_variants(source_row, variants)
    output = tmp_path / "variants.parquet"

    write_rows(output, [updated], source_schema)
    loaded = pq.read_table(output).to_pylist()[0]

    assert loaded["reward_model"]["ground_truth"] == variants
    assert loaded["extra_info"]["answer_variants"] == variants
    assert pq.read_schema(output).field("index").type == source_schema.field("index").type
