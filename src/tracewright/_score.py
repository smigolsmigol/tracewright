"""Scorers compare a candidate replay output against the recorded baseline.

V0 shipped exact match. V0.1 adds Pydantic-model equivalence (load schema,
validate both sides, compare). V0.2 will add embedding cosine + LLM-judge.
The Scorer Protocol stays narrow on purpose so plug-ins can land without
churning the engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from tracewright._models import ScoreResult

__all__ = [
    "ExactMatchScorer",
    "PydanticEquivalenceScorer",
    "ScoreResult",
    "Scorer",
    "ToolCall",
    "ToolCallDivergence",
    "tool_call_divergence",
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


@dataclass(frozen=True)
class ToolCall:
    """One tool call observed during an agent run.

    Pull from f3dx-rt's enriched JSONL trace (capture_messages=True) by
    walking the assistant messages and extracting their tool_calls field,
    or from a pydantic-ai Agent.iter() result by reading each step's
    parts where part.kind == "tool-call".
    """

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallDivergence:
    """Per-case tool-call divergence breakdown.

    name_jaccard:   Jaccard similarity of the two tool-name multisets.
                    1.0 = identical names + counts; 0.0 = disjoint.
    cardinality:    candidate count - baseline count. Positive = candidate
                    is calling more tools; negative = fewer.
    arg_match_rate: fraction of paired calls (matched in order, by name)
                    where the arguments dict is equal. None when no calls
                    pair up (one side is empty).
    matched_pairs:  list of (baseline_index, candidate_index) for the
                    name-aligned pairs the arg-match check used.
    """

    name_jaccard: float
    cardinality: int
    arg_match_rate: float | None
    matched_pairs: list[tuple[int, int]]


def tool_call_divergence(
    baseline: list[ToolCall],
    candidate: list[ToolCall],
) -> ScoreResult:
    """Score how much a replay's tool calls diverged from the recorded baseline.

    Compares two ordered lists of tool calls and returns a ScoreResult with
    a ToolCallDivergence in metadata. The headline score is name_jaccard
    weighted by arg-match fraction so an exact replay scores 1.0, a same-
    tools-different-args run scores ~0.5, and a different-tools run scores
    near 0.

    name_jaccard treats the lists as multisets - two calls to the same
    tool count separately. Arguments compare as dicts (so key order does
    not matter); for arg-match-rate, pairs are aligned greedily in order
    by name.

    Empty + empty = perfect (1.0). Either side empty when the other is
    not = 0.0.
    """
    if not baseline and not candidate:
        return ScoreResult(
            scorer="tool_call_divergence",
            passed=True,
            score=1.0,
            detail="both sides empty",
            metadata={
                "name_jaccard": 1.0,
                "cardinality": 0,
                "arg_match_rate": None,
                "matched_pairs": [],
            },
        )
    if not baseline or not candidate:
        return ScoreResult(
            scorer="tool_call_divergence",
            passed=False,
            score=0.0,
            detail=f"baseline={len(baseline)} calls, candidate={len(candidate)} calls",
            metadata={
                "name_jaccard": 0.0,
                "cardinality": len(candidate) - len(baseline),
                "arg_match_rate": None,
                "matched_pairs": [],
            },
        )

    name_jaccard = _multiset_jaccard(
        [c.name for c in baseline],
        [c.name for c in candidate],
    )

    matched_pairs = _greedy_pair_by_name(baseline, candidate)
    if matched_pairs:
        arg_matches = sum(
            1 for bi, ci in matched_pairs if baseline[bi].arguments == candidate[ci].arguments
        )
        arg_match_rate: float | None = arg_matches / len(matched_pairs)
    else:
        arg_match_rate = None

    headline = name_jaccard if arg_match_rate is None else name_jaccard * arg_match_rate

    return ScoreResult(
        scorer="tool_call_divergence",
        passed=headline >= 0.999,
        score=headline,
        detail=None
        if headline >= 0.999
        else (
            f"name_jaccard={name_jaccard:.3f} "
            f"arg_match_rate={arg_match_rate if arg_match_rate is None else f'{arg_match_rate:.3f}'} "
            f"cardinality={len(candidate) - len(baseline):+d}"
        ),
        metadata={
            "name_jaccard": name_jaccard,
            "cardinality": len(candidate) - len(baseline),
            "arg_match_rate": arg_match_rate,
            "matched_pairs": matched_pairs,
        },
    )


def _multiset_jaccard(a: list[str], b: list[str]) -> float:
    """Multiset Jaccard: |intersection| / |union| treating duplicates as distinct."""
    if not a and not b:
        return 1.0
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    i = j = 0
    intersection = 0
    while i < len(a_sorted) and j < len(b_sorted):
        if a_sorted[i] == b_sorted[j]:
            intersection += 1
            i += 1
            j += 1
        elif a_sorted[i] < b_sorted[j]:
            i += 1
        else:
            j += 1
    union = len(a_sorted) + len(b_sorted) - intersection
    return intersection / union if union > 0 else 1.0


def _greedy_pair_by_name(
    baseline: list[ToolCall],
    candidate: list[ToolCall],
) -> list[tuple[int, int]]:
    """Pair calls by name in order: for each baseline call, take the next
    unmatched candidate call with the same name. Captures the natural
    'agent did the same things in the same order' alignment."""
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for bi, b in enumerate(baseline):
        for ci, c in enumerate(candidate):
            if ci in used:
                continue
            if c.name == b.name:
                pairs.append((bi, ci))
                used.add(ci)
                break
    return pairs
