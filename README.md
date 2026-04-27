# tracewright

Replay-driven eval for f3dx and pydantic-ai traces. Take a JSONL trace, hold the input distribution fixed, swap the model or prompt, get a per-case diff. The artifact your runtime already emits becomes the regression suite.

```bash
pip install tracewright
```

```python
from tracewright import ReplayEngine, parse_jsonl
from tracewright._parse import filter_replayable

def my_candidate(case):
    # case.prompt, case.system_prompt, case.baseline_output available
    return run_my_agent(case.system_prompt, case.prompt)

rows = filter_replayable(parse_jsonl("traces.jsonl"))
engine = ReplayEngine(candidate_fn=my_candidate, candidate_model="claude-haiku-4")
for result in engine.replay_many(rows):
    if not result.all_passed:
        print(f"divergence: {result.case.prompt[:60]} -> {result.candidate_output[:60]}")
```

Read pydantic-ai logfire-shaped traces with the same engine:

```python
from tracewright import ReplayEngine, parse_pydantic_ai_jsonl, PydanticEquivalenceScorer
from pydantic import BaseModel

class Answer(BaseModel):
    intent: str
    confidence: float

rows = parse_pydantic_ai_jsonl("logfire_export.jsonl")
engine = ReplayEngine(
    candidate_fn=my_candidate,
    candidate_model="claude-haiku-4",
    scorers=[PydanticEquivalenceScorer(Answer)],   # validate both sides + compare
)
```

```bash
tracewright replay traces.jsonl --candidate myapp.replay:my_candidate \
    --candidate-model claude-haiku-4 -v
```

## Why

Agent evals are run-once snapshots today (Liu et al. 2024 "AgentBench" arXiv:2308.03688; Yang et al. 2024 "SWE-bench" arXiv:2310.06770). There's no standard way to hold the input distribution fixed and swap the model. Trivedi et al. 2024 ("Toolformer revisited" arXiv:2403.04746) names trace-replay as a missing primitive. The OpenTelemetry GenAI semconv working group defines spans but no replay tooling.

f3dx is the only Rust runtime emitting Logfire-shaped JSONL natively, and pydantic-ai with logfire enabled writes the same gen_ai.* span shape. Tracewright is the regression-suite layer that converts trace volume into a feedback loop.

## Architecture

```
tracewright/
  src/tracewright/
    _models.py      TraceRow, ReplayCase, ReplayResult, ScoreResult, Message
    _parse.py       parse_jsonl(), filter_replayable()
    _score.py       Scorer Protocol, ExactMatchScorer
    _replay.py      ReplayEngine (parse -> case -> candidate_fn -> score)
    cli.py          tracewright replay <trace.jsonl> --candidate <import:fn>
  tests/
    fixtures/enriched_trace.jsonl    sample 4-row trace
    test_replay.py                    end-to-end engine + parser tests
```

## Required trace shape

The replay engine needs three fields beyond the f3dx metadata schema: `prompt` (str), `system_prompt` (str | None), `output` (str). Today f3dx-rt's JSONL sink emits metadata only — the next f3dx push will add an opt-in `capture_messages=True` flag that includes the full prompt/output. pydantic-ai with logfire emits the equivalent in `gen_ai.input.messages` / `gen_ai.output.messages` attributes; a parser adapter for that shape is on the V0.1 list.

## Layout

```
src/tracewright/      core library
tests/                pytest suite + fixtures
pyproject.toml        hatch build, optional [pydantic-ai] extra
```

## What's not here yet

- Embedding-cosine scorer
- LLM-judge scorer
- HTML side-by-side report
- Cost + latency rollups
- CI fail-the-build mode (`--budget tokens=+5%,latency_p95=+10%`)
- Tool-call divergence reporting

## License

MIT.
