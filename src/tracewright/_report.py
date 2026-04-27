"""Aggregate replay results into a Report + render HTML / compute rollups.

The Report is the artifact a user looks at after a replay run. It carries
the per-case ReplayResults, latency stats baseline-vs-candidate, and aggregate
scorer pass/fail counts. HTML rendering is single-file self-contained — no
external CSS or JS, no build step. Open the file in any browser, get a
side-by-side diff per case.
"""

from __future__ import annotations

import html as _html
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tracewright._models import ReplayResult


@dataclass
class LatencyStats:
    """Latency rollup for one side (baseline or candidate)."""

    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> LatencyStats:
        if not samples:
            return cls(n=0, p50_ms=0.0, p95_ms=0.0, mean_ms=0.0)
        s = sorted(samples)
        return cls(
            n=len(s),
            p50_ms=_pct(s, 50),
            p95_ms=_pct(s, 95),
            mean_ms=statistics.fmean(s),
        )


@dataclass
class ScorerSummary:
    """Aggregate pass/fail for one scorer across all cases."""

    name: str
    passed: int
    failed: int
    mean_score: float

    @property
    def total(self) -> int:
        return self.passed + self.failed


@dataclass
class TokenStats:
    """Aggregate token usage across all replay cases on one side."""

    n: int
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Report:
    """The post-replay artifact: results, rollups, summary stats."""

    results: list[ReplayResult]
    candidate_model: str
    baseline_latency: LatencyStats = field(default_factory=lambda: LatencyStats(0, 0, 0, 0))
    candidate_latency: LatencyStats = field(default_factory=lambda: LatencyStats(0, 0, 0, 0))
    baseline_tokens: TokenStats = field(default_factory=lambda: TokenStats(0, 0, 0))
    scorer_summaries: list[ScorerSummary] = field(default_factory=list)

    @classmethod
    def from_results(
        cls, results: Iterable[ReplayResult], candidate_model: str
    ) -> Report:
        results_list = list(results)
        baseline_samples = [
            r.case.trace_row.duration_ms for r in results_list if r.case.trace_row.duration_ms
        ]
        candidate_samples = [r.duration_ms for r in results_list]

        baseline_tokens_in = sum(r.case.trace_row.input_tokens for r in results_list)
        baseline_tokens_out = sum(r.case.trace_row.output_tokens for r in results_list)

        scorer_names: list[str] = []
        scorer_buckets: dict[str, list[bool]] = {}
        scorer_scores: dict[str, list[float]] = {}
        for r in results_list:
            for s in r.scores:
                if s.scorer not in scorer_buckets:
                    scorer_names.append(s.scorer)
                    scorer_buckets[s.scorer] = []
                    scorer_scores[s.scorer] = []
                scorer_buckets[s.scorer].append(s.passed)
                scorer_scores[s.scorer].append(s.score)

        summaries = [
            ScorerSummary(
                name=name,
                passed=sum(scorer_buckets[name]),
                failed=len(scorer_buckets[name]) - sum(scorer_buckets[name]),
                mean_score=statistics.fmean(scorer_scores[name]) if scorer_scores[name] else 0.0,
            )
            for name in scorer_names
        ]

        return cls(
            results=results_list,
            candidate_model=candidate_model,
            baseline_latency=LatencyStats.from_samples(baseline_samples),
            candidate_latency=LatencyStats.from_samples(candidate_samples),
            baseline_tokens=TokenStats(
                n=len(results_list),
                input_tokens=baseline_tokens_in,
                output_tokens=baseline_tokens_out,
            ),
            scorer_summaries=summaries,
        )

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> int:
        return sum(1 for r in self.results if r.all_passed)

    @property
    def latency_p95_delta_pct(self) -> float:
        """Candidate p95 latency vs baseline, as percent. +10.0 means 10% slower."""
        if self.baseline_latency.p95_ms == 0:
            return 0.0
        return (
            (self.candidate_latency.p95_ms - self.baseline_latency.p95_ms)
            / self.baseline_latency.p95_ms
            * 100.0
        )

    def to_html(self) -> str:
        """Render the full report as a single self-contained HTML document."""
        return _render_html(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary, useful for CI assertions + log shipping."""
        return {
            "candidate_model": self.candidate_model,
            "total_cases": self.total_cases,
            "passed_cases": self.all_passed,
            "baseline_latency": {
                "n": self.baseline_latency.n,
                "p50_ms": self.baseline_latency.p50_ms,
                "p95_ms": self.baseline_latency.p95_ms,
                "mean_ms": self.baseline_latency.mean_ms,
            },
            "candidate_latency": {
                "n": self.candidate_latency.n,
                "p50_ms": self.candidate_latency.p50_ms,
                "p95_ms": self.candidate_latency.p95_ms,
                "mean_ms": self.candidate_latency.mean_ms,
            },
            "latency_p95_delta_pct": self.latency_p95_delta_pct,
            "baseline_tokens": {
                "input": self.baseline_tokens.input_tokens,
                "output": self.baseline_tokens.output_tokens,
                "total": self.baseline_tokens.total_tokens,
            },
            "scorers": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "failed": s.failed,
                    "mean_score": s.mean_score,
                }
                for s in self.scorer_summaries
            ],
        }


def _pct(sorted_samples: list[float], pct: int) -> float:
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    k = (pct / 100.0) * (len(sorted_samples) - 1)
    f = int(k)
    c = min(f + 1, len(sorted_samples) - 1)
    return sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f)


_HTML_CSS = """
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0; padding: 24px; background: #0d1117; color: #c9d1d9; }
h1 { margin: 0 0 8px; font-size: 20px; }
h2 { margin: 24px 0 8px; font-size: 16px; color: #58a6ff; }
.muted { color: #8b949e; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 12px 0 24px; }
.stat { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
.stat-label { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-value { font-size: 20px; font-weight: 600; margin-top: 4px; }
.pass { color: #3fb950; }
.fail { color: #f85149; }
.delta-bad { color: #f85149; }
.delta-good { color: #3fb950; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 16px; }
th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #30363d; }
th { color: #8b949e; font-weight: 500; }
.case { background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin: 12px 0; overflow: hidden; }
.case-head { padding: 12px 16px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
.case-head .marker { font-weight: 600; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
.case-head .marker.pass { background: #1a3a23; }
.case-head .marker.fail { background: #3a1a1a; }
.case-body { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
.col { padding: 12px 16px; min-width: 0; }
.col + .col { border-left: 1px solid #30363d; }
.col-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
pre { background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 8px 12px; margin: 0; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font: 12px/1.5 ui-monospace, "Cascadia Code", "JetBrains Mono", Consolas, monospace; }
.scorer-row { font-size: 12px; padding: 4px 0; }
.detail { color: #8b949e; font-size: 11px; margin-top: 4px; }
.prompt { background: #0d1117; border-left: 3px solid #58a6ff; padding: 8px 12px; margin: 8px 16px 12px; font-family: ui-monospace, monospace; font-size: 12px; }
"""


def _render_html(report: Report) -> str:
    rows = "\n".join(_render_case(i, r) for i, r in enumerate(report.results))
    delta = report.latency_p95_delta_pct
    delta_class = "delta-good" if delta <= 0 else "delta-bad"
    delta_str = f"{delta:+.1f}%"

    scorer_rows = "\n".join(
        f"<tr><td>{_html.escape(s.name)}</td>"
        f"<td><span class='pass'>{s.passed}</span> / <span class='fail'>{s.failed}</span></td>"
        f"<td>{s.mean_score:.2f}</td></tr>"
        for s in report.scorer_summaries
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>tracewright report</title><style>{_HTML_CSS}</style></head>
<body>
  <h1>tracewright replay report</h1>
  <div class="muted">candidate: <code>{_html.escape(report.candidate_model)}</code> · {report.total_cases} cases · {report.all_passed} all-passed</div>

  <div class="summary">
    <div class="stat"><div class="stat-label">Cases</div><div class="stat-value">{report.total_cases}</div></div>
    <div class="stat"><div class="stat-label">All-passed</div><div class="stat-value pass">{report.all_passed}</div></div>
    <div class="stat"><div class="stat-label">Failures</div><div class="stat-value fail">{report.total_cases - report.all_passed}</div></div>
    <div class="stat"><div class="stat-label">Baseline p95</div><div class="stat-value">{report.baseline_latency.p95_ms:.1f}ms</div></div>
    <div class="stat"><div class="stat-label">Candidate p95</div><div class="stat-value">{report.candidate_latency.p95_ms:.1f}ms</div></div>
    <div class="stat"><div class="stat-label">p95 delta</div><div class="stat-value {delta_class}">{delta_str}</div></div>
  </div>

  <h2>Scorers</h2>
  <table><thead><tr><th>Scorer</th><th>Pass / Fail</th><th>Mean score</th></tr></thead>
  <tbody>{scorer_rows or '<tr><td colspan=3 class=muted>no scorers</td></tr>'}</tbody></table>

  <h2>Cases</h2>
  {rows}
</body></html>"""


def _render_case(i: int, r: ReplayResult) -> str:
    marker_cls = "pass" if r.all_passed else "fail"
    marker_txt = "PASS" if r.all_passed else "FAIL"
    prompt = _html.escape(r.case.prompt[:500])
    baseline = _html.escape(r.case.baseline_output)
    candidate = _html.escape(r.candidate_output)
    score_rows = "\n".join(
        f"<div class='scorer-row'>"
        f"<span class='{'pass' if s.passed else 'fail'}'>{'✓' if s.passed else '✗'}</span> "
        f"{_html.escape(s.scorer)} = {s.score:.2f}"
        + (f"<div class='detail'>{_html.escape(s.detail)}</div>" if s.detail else "")
        + "</div>"
        for s in r.scores
    )
    baseline_dur = r.case.trace_row.duration_ms
    return f"""
  <div class="case">
    <div class="case-head">
      <div>case {i}: <span class="muted">{r.duration_ms:.1f}ms vs baseline {baseline_dur:.1f}ms</span></div>
      <div class="marker {marker_cls}">{marker_txt}</div>
    </div>
    <div class="prompt">{prompt}</div>
    <div class="case-body">
      <div class="col">
        <div class="col-label">baseline</div>
        <pre>{baseline}</pre>
        <div class="detail">{r.case.baseline_model or ''}</div>
      </div>
      <div class="col">
        <div class="col-label">candidate</div>
        <pre>{candidate}</pre>
        <div class="detail">{_html.escape(r.candidate_model)}</div>
      </div>
    </div>
    <div style="padding: 8px 16px; border-top: 1px solid #30363d;">
      {score_rows}
    </div>
  </div>"""
