"""Adapter from tracewright TraceRows to pydantic_evals.Dataset.

The right architecture: tracewright produces the trace-replay shape; pydantic-evals
owns the evaluator + reporting shape (LLMJudge, Equals, IsInstance, embedding-cosine
custom evaluators, markdown reports, async retries). This module bridges them so
existing tracewright JSONL traces feed straight into the pydantic-evals workflow.

Usage:

    from tracewright import to_pydantic_evals_dataset
    from pydantic_evals.evaluators import LLMJudge, Equals

    dataset = to_pydantic_evals_dataset(
        "traces.jsonl",
        evaluators=(Equals(), LLMJudge(rubric="answer is correct")),
    )
    report = await dataset.evaluate(my_candidate_fn)

The candidate fn that pydantic-evals expects is `(inputs) -> output`. We map
TraceRow.prompt to inputs and TraceRow.output to expected_output. TraceRow itself
lands in case.metadata so evaluators can read system_prompt, model, tokens, etc.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

try:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator
except ImportError as e:
    raise ImportError(
        "tracewright.to_pydantic_evals_dataset requires pydantic-evals. "
        "Install with: pip install tracewright[pydantic-evals]"
    ) from e

from tracewright._models import TraceRow
from tracewright._parse import filter_replayable, parse_jsonl
from tracewright._pydantic_ai import parse_pydantic_ai_jsonl


def to_pydantic_evals_dataset(
    source: str | Path | Iterable[TraceRow],
    *,
    name: str | None = None,
    case_name_template: str = "trace-{ts}",
    evaluators: tuple[Evaluator, ...] = (),
    pydantic_ai_logfire: bool = False,
) -> Dataset:
    """Build a pydantic_evals.Dataset from a JSONL trace.

    `source` accepts a path to an f3dx-shaped or pydantic-ai-logfire-shaped
    JSONL file, or an iterable of TraceRow records. When pydantic_ai_logfire
    is True and source is a path, the OTel-span parser is used; otherwise
    the f3dx JSONL parser.

    `evaluators` forwards to pydantic_evals.Dataset; pass any of its built-in
    evaluators (Equals, IsInstance, Contains, Python, MaxDuration, LLMJudge)
    or custom Evaluator subclasses for embedding-cosine, semantic similarity,
    schema validation, etc.

    Each yielded Case has:
        inputs           = TraceRow.prompt
        expected_output  = TraceRow.output
        metadata         = the full TraceRow (system_prompt, model, tokens, etc.)
        name             = case_name_template.format(...) using TraceRow attrs
    """
    rows: Iterable[TraceRow]
    if isinstance(source, str | Path):
        if pydantic_ai_logfire:
            rows = parse_pydantic_ai_jsonl(source)
        else:
            rows = filter_replayable(parse_jsonl(source))
    else:
        rows = source

    cases: list[Case] = []
    for i, row in enumerate(rows):
        if row.prompt is None or row.output is None:
            continue
        case_name = case_name_template.format(
            ts=row.ts,
            i=i,
            model=row.model or "unknown",
        )
        cases.append(
            Case(
                name=case_name,
                inputs=row.prompt,
                expected_output=row.output,
                metadata=row,
            )
        )
    return Dataset(name=name, cases=cases, evaluators=evaluators)
