---
id: T-002-parallel-translations
title: Cover parallel corpus translation columns
status: todo
priority: high
spec_ref: specs/v0.1.0.md#difficulty-and-translation-options
dependencies: []
updated_at: "2026-09-14T09:14:51Z"
---

# T-002-parallel-translations Cover parallel corpus translation columns

## Description

Parallel corpus support (`--latin-column`, `--translation-lang`, `--parallel-mode`) ships in
`cloze`, but only the non-interactive ignore-with-warning path is tested
(`test_cloze_non_interactive_ignores_parallel_columns_by_default`). The include path — the one
that actually writes EN/DE translation columns into the output — has no coverage, and neither
does the interactive prompt branch.

## Acceptance

- Unit tests assert that `--parallel-mode include` writes one output column per requested `--translation-lang`.
- A test covers a requested language missing from the corpus, asserting an actionable error rather than a silent empty column.
- The interactive-prompt branch is covered with a stubbed TTY/prompt.
- `poetry run ruff check`, `poetry run mypy`, and `poetry run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
