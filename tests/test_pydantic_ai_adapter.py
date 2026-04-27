"""Tests for the pydantic-ai logfire JSONL span adapter."""

from __future__ import annotations

from pathlib import Path

from tracewright import (
    ReplayCase,
    ReplayEngine,
    parse_pydantic_ai_jsonl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pydantic_ai_spans.jsonl"


def test_parser_drops_non_chat_spans() -> None:
    rows = list(parse_pydantic_ai_jsonl(FIXTURE))
    assert len(rows) == 2, "third span has no gen_ai.input/output and must be filtered"


def test_parser_extracts_prompt_system_output_model() -> None:
    rows = list(parse_pydantic_ai_jsonl(FIXTURE))
    first = rows[0]
    assert first.prompt == "what is 2+2"
    assert first.system_prompt == "you are a calculator"
    assert first.output == "4"
    assert first.model == "gpt-4"
    second = rows[1]
    assert second.prompt == "what is the capital of France"
    assert second.system_prompt is None
    assert second.output == "Paris"


def test_parser_duration_from_unix_nanos() -> None:
    rows = list(parse_pydantic_ai_jsonl(FIXTURE))
    assert rows[0].duration_ms == 150.0
    assert rows[1].duration_ms == 210.0


def test_replay_engine_consumes_pydantic_ai_traces() -> None:
    rows = parse_pydantic_ai_jsonl(FIXTURE)

    def echo(case: ReplayCase) -> str:
        return case.baseline_output

    engine = ReplayEngine(candidate_fn=echo, candidate_model="echo")
    results = list(engine.replay_many(rows))
    assert len(results) == 2
    assert all(r.all_passed for r in results)
