"""Tests for budget parsing + enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from tracewright import (
    LatencyStats,
    ReplayCase,
    ReplayEngine,
    Report,
    enforce_budgets,
    parse_budgets,
    parse_jsonl,
)
from tracewright._parse import filter_replayable

FIXTURE = Path(__file__).parent / "fixtures" / "enriched_trace.jsonl"


def _echo(case: ReplayCase) -> str:
    return case.baseline_output


def _build_report(candidate_fn=_echo) -> Report:
    rows = filter_replayable(parse_jsonl(FIXTURE))
    engine = ReplayEngine(candidate_fn=candidate_fn, candidate_model="test")
    return Report.from_results(engine.replay_many(rows), candidate_model="test")


def test_parse_empty() -> None:
    assert parse_budgets("") == []
    assert parse_budgets("  ") == []


def test_parse_single_pct() -> None:
    [c] = parse_budgets("latency_p95=+10%")
    assert c.metric == "latency_p95"
    assert c.op == "+"
    assert c.value == 10.0
    assert c.is_pct


def test_parse_multi() -> None:
    cs = parse_budgets("latency_p95=+10%,score=>=0.9")
    assert len(cs) == 2
    assert cs[1].metric == "score" and cs[1].op == ">=" and cs[1].value == 0.9


def test_parse_invalid_raises() -> None:
    with pytest.raises(ValueError, match="unparseable"):
        parse_budgets("garbage")


def test_enforce_pass_rate_passes() -> None:
    report = _build_report(_echo)
    failures = enforce_budgets(report, parse_budgets("pass_rate=>=1.0"))
    assert failures == []


def test_enforce_pass_rate_fails_when_divergent() -> None:
    def half(case):
        return case.baseline_output if case.prompt.startswith("what is 2") else "WRONG"
    report = _build_report(half)
    failures = enforce_budgets(report, parse_budgets("pass_rate=>=1.0"))
    assert len(failures) == 1
    assert "pass_rate" in failures[0].constraint.metric
    assert failures[0].actual < 1.0


def test_enforce_score_threshold() -> None:
    report = _build_report(_echo)
    failures = enforce_budgets(report, parse_budgets("score=>=0.5"))
    assert failures == []
    failures = enforce_budgets(report, parse_budgets("score=>=1.5"))
    assert len(failures) == 1


def test_enforce_unknown_metric_returns_failure() -> None:
    report = _build_report(_echo)
    failures = enforce_budgets(report, parse_budgets("nonsense=>=1.0"))
    assert len(failures) == 1
    assert "unknown metric" in failures[0].detail


def test_enforce_relative_latency_pass_when_faster() -> None:
    report = Report(
        results=[],
        candidate_model="x",
        baseline_latency=LatencyStats(n=1, p50_ms=100, p95_ms=100, mean_ms=100),
        candidate_latency=LatencyStats(n=1, p50_ms=80, p95_ms=80, mean_ms=80),
    )
    failures = enforce_budgets(report, parse_budgets("latency_p95=+10%"))
    assert failures == []


def test_enforce_relative_latency_fail_when_slower_than_budget() -> None:
    report = Report(
        results=[],
        candidate_model="x",
        baseline_latency=LatencyStats(n=1, p50_ms=100, p95_ms=100, mean_ms=100),
        candidate_latency=LatencyStats(n=1, p50_ms=130, p95_ms=130, mean_ms=130),
    )
    failures = enforce_budgets(report, parse_budgets("latency_p95=+10%"))
    assert len(failures) == 1
    assert "+30.0%" in failures[0].detail


def test_enforce_relative_latency_zero_baseline_skips() -> None:
    report = Report(
        results=[],
        candidate_model="x",
        baseline_latency=LatencyStats(n=0, p50_ms=0, p95_ms=0, mean_ms=0),
        candidate_latency=LatencyStats(n=1, p50_ms=100, p95_ms=100, mean_ms=100),
    )
    failures = enforce_budgets(report, parse_budgets("latency_p95=+10%"))
    assert failures == []
