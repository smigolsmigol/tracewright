"""tracewright command-line entry.

V0.2 surface:
    tracewright replay <trace.jsonl> --candidate <import:fn>
        [--candidate-model NAME] [--limit N]
        [--report html=PATH | --report json=PATH]
        [--budget metric=op-value,...]
        [-v]

`--candidate` is a `module.path:callable` that takes a `ReplayCase` and
returns the candidate model's output string. Resolved at runtime so the
CLI doesn't pull every framework into its dependency tree.

`--report` writes a self-contained HTML side-by-side diff or a JSON summary.
`--budget` enforces CI thresholds; non-zero exit when any constraint fails.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

from tracewright._budget import enforce_budgets, parse_budgets
from tracewright._models import ReplayCase
from tracewright._parse import filter_replayable, parse_jsonl
from tracewright._replay import ReplayEngine
from tracewright._report import Report


def _resolve_callable(spec: str) -> Callable[[ReplayCase], str]:
    if ":" not in spec:
        raise SystemExit(f"--candidate must be 'module.path:callable', got {spec!r}")
    mod_path, attr = spec.split(":", 1)
    module = importlib.import_module(mod_path)
    fn = getattr(module, attr)
    if not callable(fn):
        raise SystemExit(f"--candidate {spec!r} is not callable")
    return fn  # type: ignore[no-any-return]


def _parse_report_target(spec: str) -> tuple[str, Path]:
    """`html=path` or `json=path` -> (kind, path)."""
    if "=" not in spec:
        raise SystemExit(f"--report must be 'html=PATH' or 'json=PATH', got {spec!r}")
    kind, _, path = spec.partition("=")
    if kind not in ("html", "json"):
        raise SystemExit(f"--report kind must be 'html' or 'json', got {kind!r}")
    return kind, Path(path)


def _cmd_replay(args: argparse.Namespace) -> int:
    candidate_fn = _resolve_callable(args.candidate)
    rows = filter_replayable(parse_jsonl(Path(args.trace)))
    engine = ReplayEngine(candidate_fn=candidate_fn, candidate_model=args.candidate_model)

    results = []
    for i, result in enumerate(engine.replay_many(rows)):
        if args.limit is not None and i >= args.limit:
            break
        results.append(result)
        marker = "PASS" if result.all_passed else "FAIL"
        score_summary = ", ".join(f"{s.scorer}={s.score:.2f}" for s in result.scores)
        print(
            f"[{marker}] case {i}: {result.duration_ms:6.1f}ms  {score_summary}",
            file=sys.stderr,
        )
        if not result.all_passed and args.verbose:
            print(f"  baseline:  {result.case.baseline_output!r}", file=sys.stderr)
            print(f"  candidate: {result.candidate_output!r}", file=sys.stderr)

    report = Report.from_results(results, candidate_model=args.candidate_model)
    n_passed = report.all_passed
    n = report.total_cases

    print(file=sys.stderr)
    print(f"{n_passed}/{n} passed", file=sys.stderr)
    print(
        f"latency p95: baseline {report.baseline_latency.p95_ms:.1f}ms "
        f"-> candidate {report.candidate_latency.p95_ms:.1f}ms "
        f"({report.latency_p95_delta_pct:+.1f}%)",
        file=sys.stderr,
    )

    for spec in args.report or []:
        kind, path = _parse_report_target(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "html":
            path.write_text(report.to_html(), encoding="utf-8")
        else:
            path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"wrote {kind} report -> {path}", file=sys.stderr)

    exit_code = 0 if n_passed == n else 1

    if args.budget:
        constraints = parse_budgets(args.budget)
        failures = enforce_budgets(report, constraints)
        if failures:
            print("\nbudget violations:", file=sys.stderr)
            for f in failures:
                print(f"  [{f.constraint.expression}] {f.detail}", file=sys.stderr)
            exit_code = 2
        else:
            print(f"\nbudget OK ({len(constraints)} constraints)", file=sys.stderr)

    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tracewright")
    sub = parser.add_subparsers(dest="cmd", required=True)
    replay = sub.add_parser("replay", help="replay a JSONL trace against a candidate")
    replay.add_argument("trace", help="path to a JSONL trace file")
    replay.add_argument(
        "--candidate",
        required=True,
        help="dotted import path of candidate fn, e.g. mypkg.replay:my_candidate",
    )
    replay.add_argument(
        "--candidate-model",
        default="unknown",
        help="model name label for the report",
    )
    replay.add_argument("--limit", type=int, default=None, help="cap number of cases")
    replay.add_argument(
        "--report",
        action="append",
        metavar="KIND=PATH",
        help="write report; KIND in {html, json}. Repeatable.",
    )
    replay.add_argument(
        "--budget",
        metavar="SPEC",
        help="comma-separated budget constraints, e.g. 'latency_p95=+10%%,score=>=0.95'. "
        "Exit 2 on violation.",
    )
    replay.add_argument("-v", "--verbose", action="store_true")
    replay.set_defaults(func=_cmd_replay)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
