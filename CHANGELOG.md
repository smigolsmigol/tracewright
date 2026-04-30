# Changelog

All notable changes to tracewright are documented here. Format follows
[keep-a-changelog](https://keepachangelog.com/en/1.1.0/) 1.1.0;
project tracks [SemVer](https://semver.org/).

## [Unreleased]

### Added

- `[cache]` extra: pulls in `f3dx.cache.cached_call` so candidate
  fns wrap into a fixture file the rest of the team can replay
  without an API key.

## [0.0.8] - 2026-04-30

### Added

- `extract_tool_calls` helper for replay rows with tool spans.

## [0.0.7] - 2026-04-30

### Added

- `tool_call_divergence` scorer: compares tool call sequences
  between candidate and baseline runs, returning per-position
  match / mismatch / extra / missing diagnostics.

## [0.0.6] - 2026-04-29

First PyPI publish via OIDC Trusted Publisher.

### Added

- `release.yml` workflow on tag push, sigstore attestations on
  every wheel.

### Security

- SHA-pin every github action across CI workflows.
- gitleaks, scorecard, dependabot enabled via the org-level
  reusable workflows.
- Banned-chars sweep across the repo.

## [0.0.5] - 2026-04-28

### Added

- `to_pydantic_evals_dataset` adapter: read an f3dx or pydantic-ai
  logfire JSONL trace, get a `pydantic_evals.Dataset` you can run
  any built-in or custom `Evaluator` against. The canonical path
  for users who want pydantic-evals owning the eval-evaluator-report
  shape.

### Changed

- Repository repositioned from a self-contained replay-and-score
  tool to an adapter for `pydantic-evals`. Lightweight in-process
  scorers (`ExactMatchScorer`, `PydanticEquivalenceScorer`) still
  ship for users who prefer them.

## [0.0.4] - earlier

### Added

- `TokenStats` aggregation in replay reports.
- `--budget` CLI flag accepts `tokens_in / tokens_out / tokens_total`
  budget metrics in addition to latency + score.

## [0.0.3] - earlier

### Added

- HTML report via `--report html=PATH`: side-by-side per-case diff,
  scorer rollups, p50/p95 latency baseline-vs-candidate, single
  self-contained file (no JS, no external CSS).
- `--budget` parser: comma-separated `metric=op-value` constraints
  on `latency_p50`, `latency_p95`, `latency_mean`, `score`,
  `pass_rate`. Exit code 2 on any violation, the CI-mode gate.

## [0.0.2] - earlier

### Added

- `parse_pydantic_ai_jsonl`: read OTel logfire spans emitted by
  pydantic-ai (with `gen_ai.input.messages` and `gen_ai.output.messages`
  attributes) and flatten into `TraceRow` records the engine consumes.
- `PydanticEquivalenceScorer`: schema-validate-then-compare scoring
  for structured outputs.

## [0.0.1] - earlier

Initial scaffold: parse JSONL, replay engine skeleton,
`ExactMatchScorer`.
