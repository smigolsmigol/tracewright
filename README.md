# tracewright

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/smigolsmigol/tracewright/badge)](https://scorecard.dev/viewer/?uri=github.com/smigolsmigol/tracewright)

Trace-replay adapter for [`pydantic-evals`](https://ai.pydantic.dev/evals/). Take a JSONL trace, get a `Dataset` you can run any pydantic-evals evaluator against (`LLMJudge`, `EqualsExpected`, custom embedding-cosine - pydantic-evals owns the eval shape). The artifact your runtime already emits becomes the regression suite.

```bash
pip install tracewright
```

```python
# pip install tracewright[pydantic-evals]
import asyncio
from pydantic_evals.evaluators import EqualsExpected, LLMJudge
from tracewright import to_pydantic_evals_dataset

dataset = to_pydantic_evals_dataset(
    "traces.jsonl",
    name="prod-regression-cw43",
    evaluators=(EqualsExpected(), LLMJudge(rubric="answer is factually correct")),
)

async def my_candidate(prompt: str) -> str:
    return await run_my_agent(prompt)

report = asyncio.run(dataset.evaluate(my_candidate))
report.print()  # markdown summary, per-case pass/fail, scorer rollups
```

Same path reads pydantic-ai's native logfire-shaped JSONL spans:

```python
dataset = to_pydantic_evals_dataset(
    "logfire_export.jsonl",
    pydantic_ai_logfire=True,
    evaluators=(LLMJudge(rubric="answer is factually correct"),),
)
```

Lightweight in-process scorers (no pydantic-evals dep) still ship for users who prefer them:

```python
from tracewright import ReplayEngine, parse_jsonl, ExactMatchScorer, PydanticEquivalenceScorer
from tracewright._parse import filter_replayable

rows = filter_replayable(parse_jsonl("traces.jsonl"))
engine = ReplayEngine(candidate_fn=my_candidate, candidate_model="claude-haiku-4")
for result in engine.replay_many(rows):
    if not result.all_passed:
        print(f"divergence: {result.case.prompt[:60]} -> {result.candidate_output[:60]}")
```

```bash
tracewright replay traces.jsonl --candidate myapp.replay:my_candidate \
    --candidate-model claude-haiku-4 \
    --report html=report.html \
    --budget "pass_rate=>=1.0,latency_p95=+10%" -v
```

Drop the resulting `report.html` into a CI artifact upload step. The single self-contained file (no JS, no external CSS) renders a side-by-side diff per case + scorer rollups + p50/p95 latency baseline-vs-candidate. `--budget` is comma-separated `metric=op-value`; supported metrics are `latency_p50`, `latency_p95`, `latency_mean`, `score`, `pass_rate`. Operators `+%` / `-%` compare candidate vs baseline; `>=`, `<=`, `==` compare absolute. Exit 2 on any violation.

## Why

Agent evals are run-once snapshots today (Liu et al. 2024 "AgentBench" arXiv:2308.03688; Yang et al. 2024 "SWE-bench" arXiv:2310.06770). There's no standard way to hold the input distribution fixed and swap the model. Trivedi et al. 2024 ("Toolformer revisited" arXiv:2403.04746) names trace-replay as a missing primitive.

`pydantic-evals` already owns the eval-evaluator-report shape - built-in `LLMJudge`, `EqualsExpected`, `Contains`, `IsInstance`, `MaxDuration`, `Python`, the `Evaluator` Protocol for custom scorers, async retries, markdown reports. Tracewright deliberately does **not** reimplement any of that. It's a 50-line bridge: JSONL traces in, `pydantic_evals.Dataset` out. f3dx is the only Rust runtime emitting Logfire-shaped JSONL natively, and pydantic-ai with logfire enabled writes the same `gen_ai.*` span shape - both feed cleanly through this adapter.

## Architecture

```
tracewright/
  src/tracewright/
    _models.py        TraceRow, Message, ReplayCase, ReplayResult, ScoreResult
    _parse.py         parse_jsonl + filter_replayable for f3dx-shaped rows
    _pydantic_ai.py   parse_pydantic_ai_jsonl for OTel logfire spans
    _score.py         Scorer Protocol + ExactMatchScorer + PydanticEquivalenceScorer
    _replay.py        ReplayEngine (parse -> case -> candidate_fn -> score)
    _pydantic_evals.py to_pydantic_evals_dataset adapter (the canonical path)
    _report.py        Report aggregation + LatencyStats + self-contained HTML render
    _budget.py        --budget parser + enforcer (latency_p50/p95/mean, score, pass_rate)
    cli.py            tracewright replay <trace.jsonl> --candidate <import:fn>
                      [--report html=PATH | json=PATH] [--budget SPEC]
  tests/
    fixtures/enriched_trace.jsonl       4-row f3dx-shaped fixture
    fixtures/pydantic_ai_spans.jsonl    3-row pydantic-ai/logfire-shaped fixture
    test_replay.py                       engine + parser core
    test_pydantic_ai_adapter.py          logfire span parser tests
    test_pydantic_equivalence.py         schema-validate-then-compare scorer tests
```

## Required trace shape

The replay engine needs three fields beyond the metadata schema: `prompt` (str), `system_prompt` (str | None), `output` (str). Two sources work today out of the box:

- **f3dx**: `f3dx.configure_traces(path, capture_messages=True)` opts in to writing the enriched fields. Off by default for PII-safety. `parse_jsonl(path)` reads them.
- **pydantic-ai with logfire**: emits OTel spans with `gen_ai.input.messages` + `gen_ai.output.messages` attributes (JSON-encoded `list[ChatMessage]`). `parse_pydantic_ai_jsonl(path)` flattens those into `TraceRow` records the engine consumes the same way.

## Layout

```
src/tracewright/      core library
tests/                pytest suite + fixtures
examples/             reference candidate fns for CLI smoke + docs
pyproject.toml        hatch build, optional [pydantic-ai] extra
.github/workflows/ci.yml   ubuntu/macos/windows + py3.10/3.12 + ruff + mypy + pytest + CLI smoke
```

## What's not here yet

- Tool-call divergence reporting
- Direct ingestion of pydantic-ai `Agent.iter()` runs (today: post-run logfire JSONL only)

For embedding-cosine, LLM-judge, semantic-similarity, or any custom scoring: write a `pydantic_evals.evaluators.Evaluator` subclass and pass it via the `evaluators=` kwarg. Pydantic-evals owns that surface; tracewright deliberately doesn't compete with it.

## Sibling projects

The f3d1 ecosystem:

- [`f3dx`](https://github.com/smigolsmigol/f3dx) - Rust runtime your Python imports. Drop-in for openai + anthropic SDKs with native SSE streaming, agent loop with concurrent tool dispatch, OTel emission. `pip install f3dx`.
- [`f3dx-cache`](https://github.com/smigolsmigol/f3dx-cache) - Content-addressable LLM response cache + replay. redb + RFC 8785 JCS + BLAKE3. `pip install f3dx-cache`.
- [`pydantic-cal`](https://github.com/smigolsmigol/pydantic-cal) - Calibration metrics for `pydantic-evals`: ECE, MCE, ACE, Brier, reliability diagrams, Fisher-Rao geometry kernel. `pip install pydantic-cal`.
- [`f3dx-router`](https://github.com/smigolsmigol/f3dx-router) - In-process Rust router for LLM providers. Hedged-parallel + 429/5xx hot-swap. `pip install f3dx-router`.
- [`f3dx-bench`](https://github.com/smigolsmigol/f3dx-bench) - Public real-prod-traffic LLM benchmark dashboard. CF Worker + R2 + duckdb-wasm. [Live](https://f3dx-bench.pages.dev).
- [`llmkit`](https://github.com/smigolsmigol/llmkit) - Hosted API gateway with budget enforcement, session tracking, cost dashboards, MCP server. [llmkit.sh](https://llmkit.sh).
- [`keyguard`](https://github.com/smigolsmigol/keyguard) - Security linter for open source projects. Finds and fixes what others only report.

## License

MIT.
