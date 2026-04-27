"""tracewright — replay-driven eval over f3dx / pydantic-ai JSONL traces.

Take a recorded trace, hold the input distribution fixed, swap the model
or prompt, get a per-case diff. The artifact f3dx and pydantic-ai already
emit becomes the regression suite.
"""

from tracewright._budget import (
    BudgetConstraint,
    BudgetFailure,
    enforce_budgets,
    parse_budgets,
)
from tracewright._models import ReplayCase, ReplayResult, TraceRow
from tracewright._parse import parse_jsonl
from tracewright._pydantic_ai import parse_pydantic_ai_jsonl
from tracewright._replay import ReplayEngine
from tracewright._report import LatencyStats, Report, ScorerSummary
from tracewright._score import ExactMatchScorer, PydanticEquivalenceScorer, Scorer, ScoreResult

__all__ = [
    "BudgetConstraint",
    "BudgetFailure",
    "ExactMatchScorer",
    "LatencyStats",
    "PydanticEquivalenceScorer",
    "ReplayCase",
    "ReplayEngine",
    "ReplayResult",
    "Report",
    "ScoreResult",
    "Scorer",
    "ScorerSummary",
    "TraceRow",
    "enforce_budgets",
    "parse_budgets",
    "parse_jsonl",
    "parse_pydantic_ai_jsonl",
]
__version__ = "0.0.3"
