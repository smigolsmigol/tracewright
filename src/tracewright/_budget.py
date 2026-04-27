"""Parse + enforce CI budgets against a Report.

Budget syntax: comma-separated metric=op-value pairs where op is one of
+%, -%, <=, >=. Examples:
    latency_p95=+10%      candidate p95 must not exceed baseline p95 by >10%
    score=>=0.95          mean score across all cases must be >= 0.95
    pass_rate=>=1.0       every case must pass all scorers

When any constraint fails, `enforce_budgets` returns a list of `BudgetFailure`
records; the CLI uses that list to emit a red summary and exit non-zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tracewright._report import Report

_BUDGET_PATTERN = re.compile(
    r"""
    ^
    (?P<metric>[a-z_]+[a-z0-9_]*)        # metric name
    \s*=\s*
    (?P<op>\+|-|<=|>=|==)                 # operator
    \s*
    (?P<value>[0-9]+(?:\.[0-9]+)?)        # numeric value
    \s*
    (?P<unit>%)?                          # optional %
    $
    """,
    re.VERBOSE,
)


@dataclass
class BudgetConstraint:
    metric: str
    op: str
    value: float
    is_pct: bool

    @property
    def expression(self) -> str:
        unit = "%" if self.is_pct else ""
        return f"{self.metric}={self.op}{self.value:g}{unit}"


@dataclass
class BudgetFailure:
    constraint: BudgetConstraint
    actual: float
    detail: str


def parse_budgets(spec: str) -> list[BudgetConstraint]:
    """Parse a comma-separated budget spec into BudgetConstraint records.

    Empty input yields []. Unparseable items raise ValueError naming the
    offending fragment so the user gets a useful message in the CLI.
    """
    out: list[BudgetConstraint] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        m = _BUDGET_PATTERN.match(item)
        if not m:
            raise ValueError(
                f"unparseable budget constraint {item!r}; expected "
                "metric=op-value, e.g. latency_p95=+10% or score=>=0.95"
            )
        out.append(
            BudgetConstraint(
                metric=m.group("metric"),
                op=m.group("op"),
                value=float(m.group("value")),
                is_pct=m.group("unit") == "%",
            )
        )
    return out


def enforce_budgets(
    report: Report, budgets: list[BudgetConstraint]
) -> list[BudgetFailure]:
    """Check each constraint against the Report. Return the failures (empty if OK)."""
    failures: list[BudgetFailure] = []
    for c in budgets:
        actual = _read_metric(report, c.metric)
        if actual is None:
            failures.append(
                BudgetFailure(
                    constraint=c,
                    actual=0.0,
                    detail=f"unknown metric {c.metric!r}; "
                    "supported: latency_p50, latency_p95, latency_mean, score, pass_rate",
                )
            )
            continue
        ok, detail = _evaluate(c, actual, report)
        if not ok:
            failures.append(BudgetFailure(constraint=c, actual=actual, detail=detail))
    return failures


def _read_metric(report: Report, metric: str) -> float | None:
    if metric == "latency_p50":
        return report.candidate_latency.p50_ms
    if metric == "latency_p95":
        return report.candidate_latency.p95_ms
    if metric == "latency_mean":
        return report.candidate_latency.mean_ms
    if metric == "score":
        if not report.scorer_summaries:
            return 0.0
        return sum(s.mean_score for s in report.scorer_summaries) / len(report.scorer_summaries)
    if metric == "pass_rate":
        return report.all_passed / report.total_cases if report.total_cases else 0.0
    return None


def _evaluate(c: BudgetConstraint, actual: float, report: Report) -> tuple[bool, str]:
    """Return (passed, detail) for one constraint."""
    if c.op in ("+", "-"):
        # latency-relative: candidate vs baseline by signed percent
        baseline = _baseline_for(report, c.metric)
        if baseline == 0:
            return True, "baseline is zero, skipping relative check"
        delta_pct = (actual - baseline) / baseline * 100.0
        threshold = c.value if c.op == "+" else -c.value
        # +10% means candidate must NOT exceed baseline by more than 10%
        # -10% means candidate must NOT be slower than baseline by more than -10% (effectively requires faster)
        if c.op == "+":
            ok = delta_pct <= threshold
        else:
            ok = delta_pct >= threshold
        return ok, (
            f"candidate {c.metric}={actual:.2f} vs baseline {baseline:.2f} "
            f"({delta_pct:+.1f}%); budget {c.op}{c.value}%"
        )
    if c.op == ">=":
        return actual >= c.value, f"{c.metric}={actual:.4f}, budget >= {c.value}"
    if c.op == "<=":
        return actual <= c.value, f"{c.metric}={actual:.4f}, budget <= {c.value}"
    if c.op == "==":
        return actual == c.value, f"{c.metric}={actual:.4f}, budget == {c.value}"
    return False, f"unknown op {c.op!r}"


def _baseline_for(report: Report, metric: str) -> float:
    if metric == "latency_p50":
        return report.baseline_latency.p50_ms
    if metric == "latency_p95":
        return report.baseline_latency.p95_ms
    if metric == "latency_mean":
        return report.baseline_latency.mean_ms
    return 0.0
