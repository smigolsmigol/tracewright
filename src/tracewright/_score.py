"""Scorers compare a candidate replay output against the recorded baseline.

V0 shipped exact match. V0.1 adds Pydantic-model equivalence (load schema,
validate both sides, compare). V0.2 will add embedding cosine + LLM-judge.
The Scorer Protocol stays narrow on purpose so plug-ins can land without
churning the engine.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from tracewright._models import ScoreResult

__all__ = [
    "ExactMatchScorer",
    "PydanticEquivalenceScorer",
    "ScoreResult",
    "Scorer",
]


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


class PydanticEquivalenceScorer:
    """Validate baseline + candidate against a Pydantic schema, compare equality.

    Pass when both validate AND model_dump() is equal. Fail when either side
    fails validation (with detail naming which side + the validation error)
    or when both validate but produce different objects (with detail showing
    the diff). The right scorer for any structured-output agent: bypasses
    string-formatting noise and asserts semantic equivalence.
    """

    name = "pydantic_equivalence"

    def __init__(self, schema: type[BaseModel]) -> None:
        self.schema = schema

    def score(self, baseline: str, candidate: str) -> ScoreResult:
        b_parsed, b_err = self._validate(baseline)
        c_parsed, c_err = self._validate(candidate)

        if b_err is not None and c_err is not None:
            return self._fail(f"both sides invalid against {self.schema.__name__}")
        if b_err is not None:
            return self._fail(f"baseline invalid against {self.schema.__name__}: {b_err}")
        if c_err is not None:
            return self._fail(f"candidate invalid against {self.schema.__name__}: {c_err}")

        assert b_parsed is not None and c_parsed is not None
        b_dump = b_parsed.model_dump()
        c_dump = c_parsed.model_dump()
        if b_dump == c_dump:
            return ScoreResult(
                scorer=self.name,
                passed=True,
                score=1.0,
                metadata={"schema": self.schema.__name__},
            )
        return self._fail(f"validated but not equal: baseline={b_dump!r} candidate={c_dump!r}")

    def _validate(self, raw: str) -> tuple[BaseModel | None, str | None]:
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError as e:
            return None, f"not valid JSON: {e.msg}"
        try:
            return self.schema.model_validate(payload), None
        except ValidationError as e:
            return None, str(e)

    def _fail(self, detail: str) -> ScoreResult:
        return ScoreResult(
            scorer=self.name,
            passed=False,
            score=0.0,
            detail=detail,
            metadata={"schema": self.schema.__name__},
        )
