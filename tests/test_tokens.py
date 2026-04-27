"""Tests for token-count surface: TraceRow.input_tokens/output_tokens, Report
aggregation into TokenStats, tokens_in/tokens_out budget metrics."""

from __future__ import annotations

import json

from tracewright import (
    ReplayCase,
    ReplayEngine,
    Report,
    TokenStats,
    TraceRow,
    enforce_budgets,
    parse_budgets,
    parse_jsonl,
)


def _row(**overrides) -> TraceRow:
    base = dict(
        ts=1.0,
        duration_ms=10.0,
        iterations=1,
        tool_calls_executed=0,
        messages_count=2,
        prompt="hi",
        output="hello",
    )
    base.update(overrides)
    return TraceRow(**base)


def test_trace_row_defaults_zero_tokens() -> None:
    r = _row()
    assert r.input_tokens == 0 and r.output_tokens == 0


def test_trace_row_accepts_tokens() -> None:
    r = _row(input_tokens=42, output_tokens=7)
    assert r.input_tokens == 42 and r.output_tokens == 7


def test_parse_jsonl_carries_tokens(tmp_path) -> None:
    p = tmp_path / "tokens.jsonl"
    p.write_text(
        json.dumps({
            "ts": 1.0, "duration_ms": 10.0, "iterations": 1,
            "tool_calls_executed": 0, "messages_count": 2,
            "prompt": "hi", "output": "hello",
            "input_tokens": 100, "output_tokens": 25,
        }) + "\n",
        encoding="utf-8",
    )
    [row] = list(parse_jsonl(p))
    assert row.input_tokens == 100
    assert row.output_tokens == 25


def test_report_aggregates_baseline_tokens() -> None:
    rows = [
        _row(input_tokens=10, output_tokens=5),
        _row(input_tokens=20, output_tokens=8),
        _row(input_tokens=30, output_tokens=2),
    ]

    def echo(case: ReplayCase) -> str:
        return case.baseline_output

    engine = ReplayEngine(candidate_fn=echo, candidate_model="echo")
    results = [engine.replay_one(r) for r in rows]
    report = Report.from_results(results, candidate_model="echo")

    assert isinstance(report.baseline_tokens, TokenStats)
    assert report.baseline_tokens.n == 3
    assert report.baseline_tokens.input_tokens == 60
    assert report.baseline_tokens.output_tokens == 15
    assert report.baseline_tokens.total_tokens == 75


def test_report_to_dict_includes_baseline_tokens() -> None:
    rows = [_row(input_tokens=10, output_tokens=5)]
    engine = ReplayEngine(candidate_fn=lambda c: c.baseline_output, candidate_model="x")
    report = Report.from_results([engine.replay_one(r) for r in rows], candidate_model="x")
    payload = report.to_dict()
    assert payload["baseline_tokens"] == {"input": 10, "output": 5, "total": 15}


def test_budget_tokens_in_pass() -> None:
    rows = [_row(input_tokens=50)]
    engine = ReplayEngine(candidate_fn=lambda c: c.baseline_output, candidate_model="x")
    report = Report.from_results([engine.replay_one(r) for r in rows], candidate_model="x")
    failures = enforce_budgets(report, parse_budgets("tokens_in=<=100"))
    assert failures == []


def test_budget_tokens_in_fail() -> None:
    rows = [_row(input_tokens=200)]
    engine = ReplayEngine(candidate_fn=lambda c: c.baseline_output, candidate_model="x")
    report = Report.from_results([engine.replay_one(r) for r in rows], candidate_model="x")
    failures = enforce_budgets(report, parse_budgets("tokens_in=<=100"))
    assert len(failures) == 1
    assert failures[0].actual == 200.0


def test_budget_tokens_total_threshold() -> None:
    rows = [_row(input_tokens=400, output_tokens=150)]
    engine = ReplayEngine(candidate_fn=lambda c: c.baseline_output, candidate_model="x")
    report = Report.from_results([engine.replay_one(r) for r in rows], candidate_model="x")
    pass_failures = enforce_budgets(report, parse_budgets("tokens_total=<=600"))
    fail_failures = enforce_budgets(report, parse_budgets("tokens_total=<=500"))
    assert pass_failures == []
    assert len(fail_failures) == 1
