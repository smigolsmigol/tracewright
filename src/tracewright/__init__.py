"""tracewright — replay-driven eval over f3dx / pydantic-ai JSONL traces.

Take a recorded trace, hold the input distribution fixed, swap the model
or prompt, get a per-case diff. The artifact f3dx and pydantic-ai already
emit becomes the regression suite.
"""

from tracewright._models import ReplayCase, ReplayResult, TraceRow
from tracewright._parse import parse_jsonl
from tracewright._replay import ReplayEngine
from tracewright._score import ExactMatchScorer, Scorer, ScoreResult

__all__ = [
    "ExactMatchScorer",
    "ReplayCase",
    "ReplayEngine",
    "ReplayResult",
    "ScoreResult",
    "Scorer",
    "TraceRow",
    "parse_jsonl",
]
__version__ = "0.0.1"
