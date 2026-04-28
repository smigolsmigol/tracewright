"""tracewright - replay-driven eval over f3dx / pydantic-ai JSONL traces.

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
from tracewright._models import Message, ReplayCase, ReplayResult, TraceRow
from tracewright._parse import parse_jsonl
from tracewright._pydantic_ai import parse_pydantic_ai_jsonl
from tracewright._replay import ReplayEngine
from tracewright._report import LatencyStats, Report, ScorerSummary, TokenStats
from tracewright._score import (
    ExactMatchScorer,
    PydanticEquivalenceScorer,
    Scorer,
    ScoreResult,
    ToolCall,
    ToolCallDivergence,
    extract_tool_calls,
    tool_call_divergence,
)

__all__ = [
    "BudgetConstraint",
    "BudgetFailure",
    "ExactMatchScorer",
    "LatencyStats",
    "Message",
    "PydanticEquivalenceScorer",
    "ReplayCase",
    "ReplayEngine",
    "ReplayResult",
    "Report",
    "ScoreResult",
    "Scorer",
    "ScorerSummary",
    "TokenStats",
    "ToolCall",
    "ToolCallDivergence",
    "TraceRow",
    "enforce_budgets",
    "extract_tool_calls",
    "parse_budgets",
    "parse_jsonl",
    "parse_pydantic_ai_jsonl",
    "tool_call_divergence",
    "to_pydantic_evals_dataset",
]
__version__ = "0.0.8"


def to_pydantic_evals_dataset(*args: object, **kwargs: object) -> object:
    """Lazy import of the pydantic-evals adapter.

    Local import keeps `import tracewright` cheap when the optional
    [pydantic-evals] extra isn't installed.
    """
    from tracewright._pydantic_evals import to_pydantic_evals_dataset as _impl

    return _impl(*args, **kwargs)  # type: ignore[arg-type]
