"""Tests for the pydantic-evals Dataset adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip the whole module when pydantic-evals isn't installed.
pydantic_evals = pytest.importorskip("pydantic_evals")

from tracewright import to_pydantic_evals_dataset  # noqa: E402  (import gated by importorskip above)

FIXTURE = Path(__file__).parent / "fixtures" / "enriched_trace.jsonl"
PYDANTIC_AI_FIXTURE = Path(__file__).parent / "fixtures" / "pydantic_ai_spans.jsonl"


def test_adapter_builds_dataset_with_correct_case_count() -> None:
    dataset = to_pydantic_evals_dataset(FIXTURE)
    # Fixture has 4 rows; one lacks prompt+output and gets filtered.
    assert len(dataset.cases) == 3


def test_adapter_maps_prompt_and_output() -> None:
    dataset = to_pydantic_evals_dataset(FIXTURE)
    case = dataset.cases[0]
    assert case.inputs == "what is 2+2"
    assert case.expected_output == "4"


def test_adapter_stores_trace_row_in_metadata() -> None:
    dataset = to_pydantic_evals_dataset(FIXTURE)
    case = dataset.cases[0]
    assert case.metadata is not None
    # metadata is the TraceRow itself
    assert case.metadata.system_prompt == "you are a calculator"
    assert case.metadata.model == "gpt-4"


def test_adapter_uses_case_name_template() -> None:
    dataset = to_pydantic_evals_dataset(
        FIXTURE,
        case_name_template="trace-{i}-{model}",
    )
    assert dataset.cases[0].name == "trace-0-gpt-4"


def test_adapter_forwards_evaluators() -> None:
    from pydantic_evals.evaluators import EqualsExpected

    ev = EqualsExpected()
    dataset = to_pydantic_evals_dataset(FIXTURE, evaluators=(ev,))
    assert ev in tuple(dataset.evaluators)


def test_adapter_handles_pydantic_ai_logfire_shape() -> None:
    dataset = to_pydantic_evals_dataset(
        PYDANTIC_AI_FIXTURE,
        pydantic_ai_logfire=True,
    )
    # Fixture has 2 chat spans + 1 unrelated; the unrelated drops.
    assert len(dataset.cases) == 2
    assert dataset.cases[0].inputs == "what is 2+2"
    assert dataset.cases[1].inputs == "what is the capital of France"


def test_adapter_named_dataset() -> None:
    dataset = to_pydantic_evals_dataset(FIXTURE, name="smoke-eval")
    assert dataset.name == "smoke-eval"
