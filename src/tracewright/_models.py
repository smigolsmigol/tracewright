"""Typed records for trace rows, replay cases, and replay results.

The `TraceRow` shape is the union of fields f3dx-rt's JSONL sink emits today
plus the optional message-capture fields that the f3dx capture_messages flag
will add (next f3dx push). Tracewright works against the enriched shape; the
parser raises a clear error when the source lacks the fields it needs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """One entry in the conversation history."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class TraceRow(BaseModel):
    """One row from an f3dx JSONL trace sink.

    The unenriched f3dx schema (today) carries metadata only: ts, duration_ms,
    iterations, tool_calls_executed, etc. The enriched schema (next f3dx push,
    `capture_messages=True`) adds prompt + system_prompt + output + messages
    so a downstream replay engine can rebuild the exact request.
    """

    model_config = ConfigDict(extra="allow")

    ts: float
    duration_ms: float
    iterations: int
    tool_calls_executed: int
    messages_count: int

    # Enriched fields - required for replay; raise on missing in the parser.
    prompt: str | None = None
    system_prompt: str | None = None
    output: str | None = None
    messages: list[Message] | None = None
    model: str | None = None

    # Cost rollup inputs. f3dx-rt writes these unconditionally as of v0.0.8;
    # absent or zero on older traces. Used by Report.cost_delta_pct + the
    # tokens_in/tokens_out budget metrics.
    input_tokens: int = 0
    output_tokens: int = 0


class ReplayCase(BaseModel):
    """One case to replay: prompt + system + tool calls in original order, plus the
    recorded baseline output to diff against."""

    trace_row: TraceRow
    prompt: str
    system_prompt: str | None = None
    baseline_output: str
    baseline_model: str | None = None


class ScoreResult(BaseModel):
    """Output of one scorer on one replay."""

    scorer: str
    passed: bool
    score: float
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayResult(BaseModel):
    """Result of replaying one case against a candidate model."""

    case: ReplayCase
    candidate_output: str
    candidate_model: str
    duration_ms: float
    scores: list[ScoreResult]

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.scores)
