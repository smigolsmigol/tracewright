"""Tests for Report aggregation + HTML rendering."""

from __future__ import annotations

from pathlib import Path

from tracewright import (
    LatencyStats,
    ReplayCase,
    ReplayEngine,
    Report,
    ScorerSummary,
    parse_jsonl,
)
from tracewright._parse import filter_replayable

FIXTURE = Path(__file__).parent / "fixtures" / "enriched_trace.jsonl"


def _echo(case: ReplayCase) -> str:
    return case.baseline_output


def _wrong(case: ReplayCase) -> str:
    return "WRONG"


def _replay_all(candidate_fn) -> list:
    rows = filter_replayable(parse_jsonl(FIXTURE))
    engine = ReplayEngine(candidate_fn=candidate_fn, candidate_model="test")
    return list(engine.replay_many(rows))


def test_latency_stats_empty() -> None:
    s = LatencyStats.from_samples([])
    assert s.n == 0 and s.p50_ms == 0.0 and s.p95_ms == 0.0


def test_latency_stats_single() -> None:
    s = LatencyStats.from_samples([42.0])
    assert s.n == 1
    assert s.p50_ms == 42.0
    assert s.p95_ms == 42.0


def test_latency_stats_p95() -> None:
    s = LatencyStats.from_samples([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert s.p50_ms == 5.5
    assert s.p95_ms > 9.0


def test_report_aggregates_pass_count() -> None:
    report = Report.from_results(_replay_all(_echo), candidate_model="echo")
    assert report.total_cases == 3
    assert report.all_passed == 3


def test_report_aggregates_failures() -> None:
    report = Report.from_results(_replay_all(_wrong), candidate_model="wrong")
    assert report.total_cases == 3
    assert report.all_passed == 0


def test_report_scorer_summary() -> None:
    report = Report.from_results(_replay_all(_echo), candidate_model="echo")
    assert len(report.scorer_summaries) == 1
    s = report.scorer_summaries[0]
    assert isinstance(s, ScorerSummary)
    assert s.name == "exact_match"
    assert s.passed == 3 and s.failed == 0
    assert s.mean_score == 1.0


def test_report_to_dict_serializable() -> None:
    import json
    report = Report.from_results(_replay_all(_echo), candidate_model="echo")
    blob = json.dumps(report.to_dict())
    assert "candidate_model" in blob
    assert "latency_p95_delta_pct" in blob


def test_report_html_self_contained() -> None:
    report = Report.from_results(_replay_all(_echo), candidate_model="echo")
    html = report.to_html()
    assert "<!doctype html>" in html
    assert "<style>" in html and "</style>" in html
    # Content: candidate name + at least one PASS marker per case
    assert "echo" in html
    assert html.count("PASS") >= 3
    # No external resource references — fully self-contained
    assert "src=" not in html and "href=" not in html


def test_report_html_renders_failures() -> None:
    report = Report.from_results(_replay_all(_wrong), candidate_model="wrong")
    html = report.to_html()
    assert html.count("FAIL") >= 3
    assert "WRONG" in html  # candidate output rendered


def test_latency_p95_delta_pct_zero_when_baseline_zero() -> None:
    report = Report(
        results=[],
        candidate_model="x",
        baseline_latency=LatencyStats(n=0, p50_ms=0, p95_ms=0, mean_ms=0),
        candidate_latency=LatencyStats(n=1, p50_ms=10, p95_ms=10, mean_ms=10),
    )
    assert report.latency_p95_delta_pct == 0.0
