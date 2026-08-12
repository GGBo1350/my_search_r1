"""Schema-driven parser for batched tool calls.

Qwen3 emits tool calls as JSON objects wrapped in ``<tool_call>`` blocks.
A parallel batch packs every call of a turn into a single block with one JSON
object per line; the parser returns all of them so ``ToolAgentLoop`` executes
the batch concurrently.  The older nested XML format and the single-JSON-per-
block format remain accepted so existing trajectories can still be evaluated.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional
from uuid import uuid4

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _decode_json_objects(body: str) -> tuple[list[dict], bool]:
    """Decode one or more consecutive JSON objects from a tool-call block body.

    A parallel batch packs every call of a turn into a single
    ``<tool_call>...</tool_call>`` block with one JSON object per line.  The
    block is valid only when every object decodes and only whitespace remains.
    """
    decoder = json.JSONDecoder()
    objects: list[dict] = []
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


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_regions(text: str) -> str:
    """Blank out <think>...</think> blocks so tool-call syntax mentioned only
    in reasoning is never treated as a real action."""
    return _THINK_RE.sub(" ", text)


@ToolParser.register("search_r1_v3")
class BatchedXMLToolParser(ToolParser):
    """Parse one or more schema-defined calls from a single assistant turn."""

    tool_calls_pattern = re.compile(r"<tool_calls>\s*(.*?)\s*</tool_calls>", re.DOTALL)
    native_tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    @property
    def stop_strings(self) -> list[str]:
        return ["</tool_call>"]

    @rollout_trace_op
    async def extract_tool_calls(
        self,
        responses_ids: list[int],
        tools: Optional[list[OpenAIFunctionToolSchema]] = None,
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        schemas = self._tool_schemas(tools or [])
        if not schemas:
            return text, []

        # Tool-call syntax inside <think> is planning text, not an action.
        # Parse on a think-stripped copy so such mentions are never executed,
        # while still returning the original text to the agent loop.
        action_text = _strip_think_regions(text)
        group_match = self.tool_calls_pattern.search(action_text)
        if group_match:
            calls = self._parse_group(group_match.group(1), schemas)
            if calls:
                return text, calls

        native_calls = self._parse_native_calls(action_text, schemas)
        if native_calls:
            return text, native_calls

        attempted = self._first_attempted_tool(action_text, schemas)
        if "<tool_calls>" in action_text or "<tool_call>" in action_text or attempted:
            # Dispatch a single empty call so the tool returns format guidance.
            # The trajectory reward still marks the XML batch malformed.
            return text, [self._make_call(attempted or next(iter(schemas)), {})]
        return text, []

    @staticmethod
    def _tool_schemas(tools: list[OpenAIFunctionToolSchema]) -> dict[str, set[str]]:
        schemas: dict[str, set[str]] = {}
        for tool in tools:
            if tool.type != "function":
                continue
            properties = tool.function.parameters.properties if tool.function.parameters else {}
            schemas[tool.function.name] = set(properties)
        return schemas

    def _parse_group(self, body: str, schemas: dict[str, set[str]]) -> list[FunctionCall]:
        matched: list[tuple[int, FunctionCall]] = []
        for tool_name, valid_parameters in schemas.items():
            call_pattern = re.compile(
                rf"<{re.escape(tool_name)}>\s*(.*?)\s*</{re.escape(tool_name)}>",
                re.DOTALL,
            )
            for call_match in call_pattern.finditer(body):
                parameters = self._extract_parameters(call_match.group(1), valid_parameters)
                if parameters:
                    matched.append((call_match.start(), self._make_call(tool_name, parameters)))
        matched.sort(key=lambda item: item[0])
        return [call for _, call in matched]

    def _parse_native_calls(
        self,
        text: str,
        schemas: dict[str, set[str]],
    ) -> list[FunctionCall]:
        calls: list[FunctionCall] = []
        for match in self.native_tool_call_pattern.finditer(text):
            payloads, valid = _decode_json_objects(match.group(1))
            if not valid:
                logger.warning("Ignoring malformed native tool call batch: %s", match.group(1)[:200])
                continue
            for payload in payloads:
                name = payload.get("name")
                arguments = payload.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (AttributeError, json.JSONDecodeError, TypeError):
                        logger.warning("Ignoring malformed native tool call: %s", match.group(1)[:200])
                        continue
                if name not in schemas or not isinstance(arguments, dict):
                    continue
                filtered = {key: value for key, value in arguments.items() if key in schemas[name]}
                if filtered:
                    calls.append(self._make_call(name, filtered))
        return calls

    @staticmethod
    def _extract_parameters(body: str, valid_parameters: set[str]) -> dict[str, str]:
        parameters = {}
        for parameter in valid_parameters:
            match = re.search(
                rf"<{re.escape(parameter)}>\s*(.*?)\s*</{re.escape(parameter)}>",
                body,
                re.DOTALL,
            )
            if match and match.group(1).strip():
                parameters[parameter] = match.group(1).strip()
        return parameters

    @staticmethod
    def _first_attempted_tool(text: str, schemas: dict[str, set[str]]) -> str | None:
        attempts = []
        for tool_name in schemas:
            position = text.find(f"<{tool_name}>")
            if position >= 0:
                attempts.append((position, tool_name))
        return min(attempts)[1] if attempts else None

    @staticmethod
    def _make_call(name: str, parameters: dict[str, object]) -> FunctionCall:
        arguments = json.dumps(parameters, ensure_ascii=False)
        # verl v0.9 added ``tool_call_id`` to FunctionCall.  The v0.8 model
        # ignores unknown Pydantic fields, so branch explicitly and keep this
        # parser usable with both the Torch 2.8/v0.8 stack and newer verl.
        if "tool_call_id" not in getattr(FunctionCall, "model_fields", {}):
            return FunctionCall(name=name, arguments=arguments)
        return FunctionCall(
            name=name,
            arguments=arguments,
            tool_call_id=f"call_{uuid4().hex}",
        )
