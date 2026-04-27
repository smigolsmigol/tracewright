"""Scorers compare a candidate replay output against the recorded baseline.

V0 ships exact match. V0.1 will add Pydantic-model equivalence, embedding
cosine, LLM-judge. The Scorer Protocol stays narrow on purpose so plug-ins
can land without churning the engine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tracewright._models import ScoreResult


@runtime_checkable
class Scorer(Protocol):
    """Protocol every scorer implements. Score one (baseline, candidate) pair."""

    name: str

    def score(self, baseline: str, candidate: str) -> ScoreResult: ...


class ExactMatchScorer:
    """Strict string equality after stripping leading/trailing whitespace."""

    name = "exact_match"

    def score(self, baseline: str, candidate: str) -> ScoreResult:
        b = baseline.strip()
        c = candidate.strip()
        passed = b == c
        return ScoreResult(
            scorer=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            detail=None if passed else f"baseline ({len(b)} chars) != candidate ({len(c)} chars)",
        )
