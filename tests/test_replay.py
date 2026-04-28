"""End-to-end replay test: parse the fixture, replay against a stub candidate,
verify exact-match scoring catches divergence."""

from __future__ import annotations

from pathlib import Path

from tracewright import (
    ExactMatchScorer,
    ReplayCase,
    ReplayEngine,
    parse_jsonl,
)
from tracewright._parse import filter_replayable

FIXTURE = Path(__file__).parent / "fixtures" / "enriched_trace.jsonl"


def perfect_candidate(case: ReplayCase) -> str:
    """Echoes the baseline - every case passes exact-match."""
    return case.baseline_output


def wrong_candidate(case: ReplayCase) -> str:
    """Always returns the wrong answer - every case fails exact-match."""
    return "WRONG"


def test_parse_drops_unenriched_rows() -> None:
    rows = list(parse_jsonl(FIXTURE))
    assert len(rows) == 4, "fixture has 4 rows"
    replayable = list(filter_replayable(rows))
    assert len(replayable) == 3, "fourth row lacks prompt+output, must be filtered"


def test_replay_perfect_candidate_all_pass() -> None:
    rows = filter_replayable(parse_jsonl(FIXTURE))
    engine = ReplayEngine(candidate_fn=perfect_candidate, candidate_model="stub")
    results = list(engine.replay_many(rows))
    assert len(results) == 3
    assert all(r.all_passed for r in results)
    assert all(r.scores[0].score == 1.0 for r in results)


def test_replay_wrong_candidate_all_fail() -> None:
    rows = filter_replayable(parse_jsonl(FIXTURE))
    engine = ReplayEngine(candidate_fn=wrong_candidate, candidate_model="stub")
    results = list(engine.replay_many(rows))
    assert len(results) == 3
    assert not any(r.all_passed for r in results)
    for r in results:
        assert r.scores[0].score == 0.0
        assert r.scores[0].detail is not None


def test_score_result_protocol() -> None:
    s = ExactMatchScorer()
    out = s.score("hello", "hello")
    assert out.passed and out.score == 1.0
    out = s.score("hello", "world")
    assert not out.passed and out.score == 0.0


def test_replay_engine_raises_on_unenriched_row() -> None:
    rows = list(parse_jsonl(FIXTURE))
    bare_row = rows[3]
    assert bare_row.prompt is None
    engine = ReplayEngine(candidate_fn=perfect_candidate, candidate_model="stub")
    try:
        engine.replay_one(bare_row)
    except ValueError as e:
        assert "capture_messages" in str(e)
    else:
        raise AssertionError("expected ValueError on unenriched row")
