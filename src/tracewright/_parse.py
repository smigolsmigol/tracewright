"""JSONL trace parser.

V0 reads the f3dx-shaped row format and produces typed `TraceRow` records.
Skips blank lines and JSON-decode failures with a row index in the error,
because partial / corrupted files are common when the producer crashed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from tracewright._models import TraceRow


def parse_jsonl(source: str | Path | Iterable[str]) -> Iterator[TraceRow]:
    """Parse a JSONL trace into TraceRow objects.

    `source` is a path-like to a `.jsonl` file or an iterable of raw line
    strings. Validation errors include the 0-based row index so corrupted
    rows can be located in the source quickly.
    """
    if isinstance(source, str | Path):
        with open(source, encoding="utf-8") as f:
            yield from _parse_lines(f)
    else:
        yield from _parse_lines(source)


def _parse_lines(lines: Iterable[str]) -> Iterator[TraceRow]:
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"row {i}: invalid JSON: {e}") from e
        try:
            yield TraceRow.model_validate(payload)
        except ValidationError as e:
            raise ValueError(f"row {i}: schema mismatch: {e}") from e


def filter_replayable(rows: Iterable[TraceRow]) -> Iterator[TraceRow]:
    """Drop rows that lack the enriched fields a replay needs (prompt + output)."""
    for row in rows:
        if row.prompt is None or row.output is None:
            continue
        yield row
