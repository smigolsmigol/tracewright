# Security Policy

## Reporting a vulnerability

Email smigolsmigol@protonmail.com. Please include enough detail to reproduce (trace shape, candidate signature, or scorer config that triggers the issue). Do not file public issues for suspected vulnerabilities.

Acknowledgement within 48 hours. Critical issues patched and released within 7 days; lower-severity within 30 days. Coordinated disclosure preferred.

## Supported versions

Only the latest released version on PyPI receives security fixes.

## Architecture notes for reviewers

tracewright is a pure-Python adapter (hatchling build, no compiled extensions). Runtime surface area:

- Reads JSONL files from disk (`parse_jsonl`, `parse_pydantic_ai_jsonl`).
- Imports a user-supplied candidate via `--candidate module:fn` and calls it.
- Optional pydantic-evals dependency for the `to_pydantic_evals_dataset` path. Without the extra installed, the in-process scorers stay self-contained.
- No outbound network IO from the library or CLI itself. Network calls only happen inside the user's candidate function (e.g. their LLM client).

The CLI does dynamic import of the user's `--candidate` argument. Treat it as code execution by the same actor who runs the CLI; never accept untrusted candidate strings.
