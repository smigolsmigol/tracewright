"""tracewright command-line entry.

V0 surface:
    tracewright replay <trace.jsonl> --candidate <import:fn>
        --candidate-model <name> [--limit N]

`--candidate` is an `module.path:callable` that takes a `ReplayCase` and
returns the candidate model's output string. Resolved at runtime so the
CLI doesn't pull every framework into its dependency tree.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from pathlib import Path

from tracewright._models import ReplayCase
from tracewright._parse import filter_replayable, parse_jsonl
from tracewright._replay import ReplayEngine


def _resolve_callable(spec: str) -> Callable[[ReplayCase], str]:
    if ":" not in spec:
        raise SystemExit(f"--candidate must be 'module.path:callable', got {spec!r}")
    mod_path, attr = spec.split(":", 1)
    module = importlib.import_module(mod_path)
    fn = getattr(module, attr)
    if not callable(fn):
        raise SystemExit(f"--candidate {spec!r} is not callable")
    return fn  # type: ignore[no-any-return]


def _cmd_replay(args: argparse.Namespace) -> int:
    candidate_fn = _resolve_callable(args.candidate)
    rows = filter_replayable(parse_jsonl(Path(args.trace)))
    engine = ReplayEngine(candidate_fn=candidate_fn, candidate_model=args.candidate_model)
    n = 0
    n_passed = 0
    for i, result in enumerate(engine.replay_many(rows)):
        if args.limit is not None and i >= args.limit:
            break
        n += 1
        if result.all_passed:
            n_passed += 1
        marker = "PASS" if result.all_passed else "FAIL"
        score_summary = ", ".join(f"{s.scorer}={s.score:.2f}" for s in result.scores)
        print(
            f"[{marker}] case {i}: {result.duration_ms:6.1f}ms  {score_summary}",
            file=sys.stderr,
        )
        if not result.all_passed and args.verbose:
            print(f"  baseline:  {result.case.baseline_output!r}", file=sys.stderr)
            print(f"  candidate: {result.candidate_output!r}", file=sys.stderr)
    print(f"\n{n_passed}/{n} passed", file=sys.stderr)
    return 0 if n_passed == n else 1


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
    replay.add_argument("-v", "--verbose", action="store_true")
    replay.set_defaults(func=_cmd_replay)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
