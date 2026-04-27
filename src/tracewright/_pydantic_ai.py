"""Adapter for pydantic-ai's logfire-shaped JSONL traces.

When `pydantic-ai` runs with logfire instrumentation, every model request emits
an OTel span whose attributes carry `gen_ai.input.messages` and
`gen_ai.output.messages` (JSON-encoded `list[ChatMessage]`). Logfire dumps
these spans as JSONL when configured with a local file sink.

This adapter parses that shape into the same `TraceRow` records that the
f3dx parser produces, so the replay engine can consume both sources without
caring which runtime emitted them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from tracewright._models import TraceRow

GEN_AI_INPUT = "gen_ai.input.messages"
GEN_AI_OUTPUT = "gen_ai.output.messages"
GEN_AI_MODEL = "gen_ai.request.model"


def parse_pydantic_ai_jsonl(source: str | Path | Iterable[str]) -> Iterator[TraceRow]:
    """Parse pydantic-ai logfire-shaped JSONL spans into TraceRow records.

    Skips spans that aren't model-request spans (no gen_ai.input/output
    attributes). Each yielded TraceRow has prompt + system_prompt + output +
    model populated, ready to feed into ReplayEngine.
    """
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as f:
            yield from _parse_lines(f)
    else:
        yield from _parse_lines(source)


def _parse_lines(lines: Iterable[str]) -> Iterator[TraceRow]:
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            span = json.loads(line)
        except json.JSONDecodeError:
            continue
        row = _span_to_trace_row(span)
        if row is not None:
            yield row


def _span_to_trace_row(span: dict[str, Any]) -> TraceRow | None:
    """Map a single OTel span dict to a TraceRow, or None if not replayable."""
    raw_attrs: Any = span.get("attributes") or {}
    attrs: dict[str, Any]
    if isinstance(raw_attrs, list):
        # Some OTel exporters emit attributes as [{key, value}, ...]; flatten.
        attrs = {item.get("key"): _otel_value(item.get("value")) for item in raw_attrs}
    else:
        attrs = raw_attrs

    raw_input = attrs.get(GEN_AI_INPUT)
    raw_output = attrs.get(GEN_AI_OUTPUT)
    if not raw_input or not raw_output:
        return None

    try:
        input_messages = json.loads(raw_input) if isinstance(raw_input, str) else raw_input
        output_messages = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    except json.JSONDecodeError:
        return None

    prompt, system_prompt = _flatten_input(input_messages)
    output = _flatten_output(output_messages)
    if prompt is None or output is None:
        return None

    return TraceRow(
        ts=span.get("start_time_unix_nano", 0) / 1e9 if "start_time_unix_nano" in span else 0.0,
        duration_ms=_duration_ms(span),
        iterations=1,
        tool_calls_executed=0,
        messages_count=len(input_messages) + 1,
        prompt=prompt,
        system_prompt=system_prompt,
        output=output,
        model=attrs.get(GEN_AI_MODEL),
    )


def _otel_value(v: Any) -> Any:
    """OTel attribute values come in tagged-union form: {'stringValue': 's'}."""
    if isinstance(v, dict):
        for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if key in v:
                return v[key]
    return v


def _duration_ms(span: dict[str, Any]) -> float:
    end = span.get("end_time_unix_nano")
    start = span.get("start_time_unix_nano")
    if not isinstance(end, (int, float)) or not isinstance(start, (int, float)):
        return 0.0
    return float(end - start) / 1e6


def _flatten_input(messages: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Pull the user prompt + system prompt out of pydantic-ai's ChatMessage list.

    pydantic-ai emits one ChatMessage per role with a parts list; user and
    system parts are TextPart {type: 'text', content: '...'}. We flatten into
    a single prompt string (last user message) + system_prompt (concatenated
    system parts).
    """
    prompt: str | None = None
    system_parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        text = _join_text_parts(msg.get("parts") or [])
        if role == "system" and text:
            system_parts.append(text)
        elif role == "user" and text:
            prompt = text
    return prompt, "\n".join(system_parts) if system_parts else None


def _flatten_output(messages: list[dict[str, Any]]) -> str | None:
    """Output is a single OutputMessage with role=assistant; concat its parts."""
    for msg in messages:
        if msg.get("role") == "assistant":
            text = _join_text_parts(msg.get("parts") or [])
            if text:
                return text
    return None


def _join_text_parts(parts: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for part in parts:
        content = part.get("content")
        if part.get("type") == "text" and isinstance(content, str):
            chunks.append(content)
    return "".join(chunks)
