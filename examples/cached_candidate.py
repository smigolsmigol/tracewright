"""Cached candidate fn for tracewright replay (uses f3dx.cache.cached_call).

Demonstrates the f3d1-wide cache-backed real-API pattern (per
smigolsmigol/f3dx docs/workflows/real_api_benches.md) applied to
tracewright candidate functions. Every candidate-side LLM call routes
through `f3dx.cache.cached_call` against a fixture file; first run
records, subsequent runs replay deterministically.

Why this matters for tracewright:
  - replay benchmarks become reproducible across machines: clone the
    repo + tracewright run, you get the same numbers without an API
    key
  - throttle exposure on long candidate runs drops to zero on re-runs
  - CI runs F3DX_BENCH_OFFLINE=1 so cache miss fails loud instead of
    silently hitting the live API

Install:
    pip install tracewright[cache]   # pulls in f3dx>=0.0.18 for f3dx.cache

Usage with tracewright CLI:
    tracewright replay traces.jsonl \\
        --candidate examples.cached_candidate:cached_openai \\
        --candidate-model gpt-4o-mini

The candidate fn signature stays the same as the rest of tracewright's
candidates -- it just internally caches the OpenAI hit.
"""
from __future__ import annotations

import os
from pathlib import Path

from tracewright import ReplayCase

# f3dx.cache is the optional dep behind the [cache] extra. Import lazily
# so the module is still importable when only base tracewright is
# installed (e.g. for type checks); the candidate fns below will raise
# at call time if the extra isn't present.
try:
    from f3dx.cache import Cache, cached_call  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover
    Cache = None  # type: ignore[assignment, misc]
    cached_call = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _fixture_path() -> Path:
    """Default fixture file alongside the repo's bench/ directory.
    Override with TRACEWRIGHT_FIXTURE env var for ad-hoc runs."""
    p = os.environ.get("TRACEWRIGHT_FIXTURE")
    if p:
        return Path(p)
    here = Path(__file__).resolve().parent
    fixture_dir = here.parent / "bench" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    return fixture_dir / "openai.redb"


def _fetch_openai(request: dict) -> dict:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(**request)
    return resp.model_dump()


def cached_openai(case: ReplayCase, *, model: str = "gpt-4o-mini") -> str:
    """Candidate that re-runs the case's prompt against `model` via OpenAI,
    routed through f3dx.cache.cached_call so the call is fixture-backed.

    First run hits OpenAI and records to bench/fixtures/openai.redb.
    Subsequent runs replay deterministically from the fixture; CI sets
    F3DX_BENCH_OFFLINE=1 so cache miss raises LookupError instead of
    silently hitting the live API.
    """
    if cached_call is None:
        raise RuntimeError(
            "tracewright[cache] not installed. "
            "Install with `pip install tracewright[cache]` to enable "
            "fixture-backed real-API candidates. "
            f"Underlying import error: {_IMPORT_ERROR}"
        )
    cache = Cache(str(_fixture_path()))
    request = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt or ""}],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    response = cached_call(cache, request, _fetch_openai, model=model)
    return response["choices"][0]["message"]["content"] or ""
