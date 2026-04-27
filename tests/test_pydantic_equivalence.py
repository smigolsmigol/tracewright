"""Tests for the PydanticEquivalenceScorer."""

from __future__ import annotations

from pydantic import BaseModel

from tracewright import PydanticEquivalenceScorer


class Answer(BaseModel):
    intent: str
    confidence: float


def test_pass_when_both_validate_and_equal() -> None:
    s = PydanticEquivalenceScorer(Answer)
    out = s.score('{"intent":"search","confidence":0.9}', '{"intent":"search","confidence":0.9}')
    assert out.passed and out.score == 1.0
    assert out.metadata["schema"] == "Answer"


def test_pass_ignores_field_order() -> None:
    s = PydanticEquivalenceScorer(Answer)
    out = s.score(
        '{"intent":"search","confidence":0.9}',
        '{"confidence":0.9,"intent":"search"}',
    )
    assert out.passed


def test_fail_when_values_differ() -> None:
    s = PydanticEquivalenceScorer(Answer)
    out = s.score('{"intent":"search","confidence":0.9}', '{"intent":"search","confidence":0.5}')
    assert not out.passed
    assert "validated but not equal" in (out.detail or "")


def test_fail_when_baseline_invalid() -> None:
    s = PydanticEquivalenceScorer(Answer)
    out = s.score('{"intent":"search"}', '{"intent":"search","confidence":0.9}')
    assert not out.passed
    assert "baseline invalid" in (out.detail or "")


def test_fail_when_candidate_invalid() -> None:
    s = PydanticEquivalenceScorer(Answer)
    out = s.score('{"intent":"search","confidence":0.9}', "not even json")
    assert not out.passed
    assert "candidate invalid" in (out.detail or "")
    assert "not valid JSON" in (out.detail or "")


def test_fail_when_both_invalid() -> None:
    s = PydanticEquivalenceScorer(Answer)
    out = s.score("not json", "also not")
    assert not out.passed
    assert "both sides invalid" in (out.detail or "")
