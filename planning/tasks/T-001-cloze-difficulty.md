---
id: T-001-cloze-difficulty
title: Cover cloze difficulty filtering
status: todo
priority: high
spec_ref: specs/v0.1.0.md#difficulty-and-translation-options
dependencies: []
updated_at: "2026-09-14T09:14:51Z"
---

# T-001-cloze-difficulty Cover cloze difficulty filtering

## Description

`--difficulty easy|medium|hard` gates which corpus verses become clozes (~15 references in
`src/latinitas_cards/cli.py`), but no test in `tests/unit/cli_test.py` exercises the option:
every cloze/preview test runs the default. A change to the scoring or the thresholds can
silently flip which verses are selected, or make the flag a no-op, without failing the suite.

## Acceptance

- Unit tests assert that each difficulty level selects a different, expected set of verses from a fixture corpus.
- A test pins the boundary behavior between adjacent levels, so a threshold change fails the suite.
- Tests run against local fixtures, no corpus download.
- `poetry run ruff check`, `poetry run mypy`, and `poetry run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
