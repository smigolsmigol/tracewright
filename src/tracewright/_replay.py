"""Replay engine.

Takes a sequence of `TraceRow` records, builds a `ReplayCase` for each,
calls a user-provided `candidate_fn` that returns the candidate model's
output for a prompt, and runs every registered scorer.

The candidate_fn signature stays minimal on purpose so the engine works
against pydantic-ai Agent.run, raw openai client, f3dx AgentRuntime, or
a plain function in tests. The harness shouldn't care which.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from tracewright._models import ReplayCase, ReplayResult, TraceRow
from tracewright._score import ExactMatchScorer, Scorer

CandidateFn = Callable[[ReplayCase], str]


@dataclass
class ReplayEngine:
    """Coordinates parse -> case-build -> candidate-call -> score across rows."""

    candidate_fn: CandidateFn
    candidate_model: str
    scorers: list[Scorer] = field(default_factory=lambda: [ExactMatchScorer()])

    def replay_one(self, row: TraceRow) -> ReplayResult:
        case = self._build_case(row)
        t0 = time.perf_counter()
        candidate_output = self.candidate_fn(case)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        scores = [s.score(case.baseline_output, candidate_output) for s in self.scorers]
        return ReplayResult(
            case=case,
            candidate_output=candidate_output,
            candidate_model=self.candidate_model,
            duration_ms=duration_ms,
            scores=scores,
        )

    def replay_many(self, rows: Iterable[TraceRow]) -> Iterator[ReplayResult]:
        for row in rows:
            yield self.replay_one(row)

    @staticmethod
    def _build_case(row: TraceRow) -> ReplayCase:
        if row.prompt is None or row.output is None:
            raise ValueError(
                "trace row lacks prompt+output - replay needs the enriched "
                "f3dx schema (configure with capture_messages=True)"
            )
        return ReplayCase(
            trace_row=row,
            prompt=row.prompt,
            system_prompt=row.system_prompt,
            baseline_output=row.output,
            baseline_model=row.model,
        )
