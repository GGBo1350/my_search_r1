from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from recipe.v2.data_preprocess import process_row, resolve_expected_tool
from recipe.v2.my_reward import compute_score


def test_preprocess_preserves_answer_aliases_and_routing_label():
    row = process_row(
        {
            "question": "法国的首都是什么？",
            "answer": ["巴黎", "Paris", "巴黎市"],
            "expected_tool": "zhihu_search",
        },
        split="train",
        index=7,
    )

    assert row["reward_model"]["ground_truth"] == ["巴黎", "Paris", "巴黎市"]
    assert row["extra_info"]["expected_tool"] == "zhihu_search"
    assert row["extra_info"]["need_tools_kwargs"] is False


def test_preprocess_uses_stable_parquet_schema_for_single_and_multiple_answers(tmp_path):
    rows = [
        process_row(
            {"question": "q1", "answer": "a1", "expected_tool": "none"},
            split="train",
            index=0,
        ),
        process_row(
            {"question": "q2", "answer": ["a2", "alias"], "expected_tool": "zhihu_search"},
            split="train",
            index=1,
        ),
    ]
    output_path = tmp_path / "mixed_answers.parquet"

    pd.DataFrame(rows).to_parquet(output_path, index=False)
    loaded = pd.read_parquet(output_path)

    assert len(loaded) == 2
    assert list(loaded.iloc[0]["reward_model"]["ground_truth"]) == ["a1"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"need_search": True}, "zhihu_search"),
        ({"need_search": False}, "none"),
        ({"expected_tool": "search"}, "zhihu_search"),
        ({"expected_tool": "calculator"}, "calculator"),
    ],
)
def test_resolve_expected_tool(raw, expected):
    assert resolve_expected_tool(raw, "none") == expected


def test_reward_distinguishes_routing_and_query_quality():
    no_tool = compute_score(
        "<think>这是常识。</think><answer>巴黎</answer>",
        ["巴黎", "Paris"],
        extra_info={"question": "法国的首都是什么？", "expected_tool": "none"},
    )
    good_search = compute_score(
        "<think>需要查询。</think>"
        "<zhihu_search><query>法国 首都</query></zhihu_search>"
        "<information>法国首都是巴黎。</information>"
        "<think>根据结果作答。</think><answer>巴黎</answer>",
        ["巴黎", "Paris"],
        extra_info={"question": "法国的首都是什么？", "expected_tool": "zhihu_search"},
    )
    bad_search = compute_score(
        "<think>需要查询。</think>"
        "<zhihu_search><query>xxxxxxxx</query></zhihu_search>"
        "<information>No results found for: xxxxxxxx</information>"
        "<think>直接猜测。</think><answer>巴黎</answer>",
        ["巴黎", "Paris"],
        extra_info={"question": "法国的首都是什么？", "expected_tool": "zhihu_search"},
    )
    unnecessary_search = compute_score(
        "<think>查询一下。</think>"
        "<zhihu_search><query>法国 首都</query></zhihu_search>"
        "<information>法国首都是巴黎。</information>"
        "<answer>巴黎</answer>",
        "巴黎",
        extra_info={"question": "法国的首都是什么？", "expected_tool": "none"},
    )

    assert no_tool["score"] == 1.25
    assert good_search["score"] == 1.5
    assert good_search["argument_valid"] is True
    assert good_search["result_valid"] is True
    assert bad_search["score"] < good_search["score"]
    assert bad_search["argument_valid"] is False
    assert bad_search["result_valid"] is False
    assert unnecessary_search["routing_correct"] is False
    assert unnecessary_search["score"] < no_tool["score"]


def test_reward_penalizes_extra_calls():
    score = compute_score(
        "<think>查询。</think>"
        "<zhihu_search><query>法国 首都</query></zhihu_search>"
        "<information>巴黎</information>"
        "<zhihu_search><query>法国 首都 城市</query></zhihu_search>"
        "<information>巴黎</information>"
        "<answer>巴黎</answer>",
        "巴黎",
        extra_info={"question": "法国的首都是什么？", "expected_tool": "zhihu_search"},
    )

    assert score["num_tool_calls"] == 2
    assert score["extra_call_penalty"] == 0.05


def test_parser_dispatches_only_first_call_in_text_order():
    from recipe.v2.my_tool_parser import SearchR1XMLToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema

    class StubTokenizer:
        @staticmethod
        def decode(value):
            return value

    schemas = [
        OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "zhihu_search",
                    "description": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ),
        OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "calculate",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ),
    ]
    parser = SearchR1XMLToolParser(StubTokenizer())
    text = (
        "<calculator><expression>2 + 2</expression></calculator><zhihu_search><query>later query</query></zhihu_search>"
    )

    _, calls = asyncio.run(parser.extract_tool_calls(text, schemas))

    assert parser.stop_token_ids == []
    assert parser.stop_strings == ["</zhihu_search>", "</calculator>"]
    assert len(calls) == 1
    assert calls[0].name == "calculator"
    assert json.loads(calls[0].arguments) == {"expression": "2 + 2"}


def test_calculator_rejects_resource_heavy_exponent():
    from recipe.v2.my_tools import calculator

    valid = asyncio.run(calculator("sqrt(16) + 2"))
    invalid = asyncio.run(calculator("2 ** (10 ** 10)"))

    assert "Result: 6.0" in valid
    assert "Calculation error" in invalid


def test_tool_agent_loop_passes_complete_stop_strings_to_vllm():
    from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

    class FakeServer:
        sampling_params = None

        async def generate(self, **kwargs):
            self.sampling_params = kwargs["sampling_params"]
            return SimpleNamespace(
                token_ids=[7],
                num_preempted=0,
                extra_fields={},
                log_probs=None,
                routed_experts=None,
            )

    loop = object.__new__(ToolAgentLoop)
    loop.tool_parser = SimpleNamespace(
        stop_token_ids=[],
        stop_strings=["</zhihu_search>", "</calculator>"],
    )
    loop.rollout_config = SimpleNamespace(name="vllm")
    loop.server_manager = FakeServer()
    loop.enable_continuous_token = False
    loop.response_length = 32
    loop.max_assistant_turns = 1
    loop.max_user_turns = 3

    agent_data = AgentData(
        messages=[],
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        metrics={},
        request_id="test-request",
        tools_kwargs={},
    )
    agent_data.prompt_ids = [1]

    state = asyncio.run(loop._handle_generating_state(agent_data, {"temperature": 0}))

    assert state is AgentState.TERMINATED
    assert loop.server_manager.sampling_params["stop"] == ["</zhihu_search>", "</calculator>"]
    assert loop.server_manager.sampling_params["include_stop_str_in_output"] is True
