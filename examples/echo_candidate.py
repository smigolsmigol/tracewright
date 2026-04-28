"""Trivial candidate fn for tracewright CLI smoke + docs.

Usage:
    tracewright replay traces.jsonl \\
        --candidate examples.echo_candidate:echo \\
        --candidate-model echo
"""

from tracewright import ReplayCase


def echo(case: ReplayCase) -> str:
    """Return the baseline verbatim - every case passes exact-match."""
    return case.baseline_output


def constant(case: ReplayCase) -> str:
    """Return a fixed string - most cases fail exact-match (sanity check)."""
    return "fixed-output"
