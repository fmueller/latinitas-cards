---
id: T-020-parse-and-validate-authored-note-import-files
title: Parse and validate authored note import files
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-import-format
dependencies: []
updated_at: "2026-10-04T09:51:24Z"
---

# T-020-parse-and-validate-authored-note-import-files Parse and validate authored note import files

## Description

Define the versioned JSONL import format for authored `vocab`, `form`, and `qa` items and a
typed loader that validates it. Source references and section labels are opaque strings;
no corpus is parsed or assumed.

## Acceptance

- The loader reads UTF-8 JSONL with a required `schema_version` and returns typed items with
  kind, key, kind-specific content, provenance (document, section, optional reference),
  status (`include`/`skip`), tags, and language tag.
- Unknown or incompatible schema versions, unknown kinds, missing required fields, and
  malformed JSON fail with the line number and failed assumption.
- Whole-file validation retains recoverable line-level errors for preview reporting,
  including errors on skipped or filtered-out rows; invalid input is never an exportable
  result. Tests cover multiple errors, including malformed JSON followed by a valid row.
- Source references are stored verbatim; tests use references from more than one corpus
  and non-corpus labels to prove nothing is interpreted.
- The format is documented with one example per kind.
- Code lives outside the legacy `cli.py` monolith.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Verification passed at 2026-10-04T09:51:24Z after Ruff, strict mypy, and
  `uv run pytest -v` (464 passed). Focused loader suite: 33 passed.
- Strict TDD: initial loader tests failed with ModuleNotFoundError, then 30 passed.
  Duplicate-key regression tests failed in all 3 cases, then all 3 passed.
- Dedicated code-simplifier made no changes. Independent General and Python
  review found no concrete findings. Security S-1 was validated by a fresh
  candidate reviewer: "Reject JSON objects with duplicate keys instead of
  silently accepting the last value." Fixed with a recursive object-pairs hook;
  fresh disposition verification marked S-1 resolved with no new issues.
- Manual loading of all documented JSONL examples returned 3 typed items with
  verbatim references from two corpora and a non-corpus lesson label.

## Implementation Notes

- 2026-10-04T09:51:24Z: verification pass
- 2026-10-04T09:51:24Z: Implemented schema-1 authored JSONL loader, typed vocab/form/qa, aggregate whole-file diagnostics and require_valid gate, opaque provenance, and per-kind format examples. Ruff/mypy/464 tests pass; review S-1 fixed via strict TDD and independently verified. See verification run timestamp recorded in this task.
