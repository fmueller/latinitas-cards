---
id: T-020-parse-and-validate-authored-note-import-files
title: Parse and validate authored note import files
status: todo
priority: high
spec_ref: specs/v0.1.1.md#authored-note-import-format
dependencies: []
updated_at: "2026-09-23T21:23:04Z"
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
- Source references are stored verbatim; tests use references from more than one corpus
  and non-corpus labels to prove nothing is interpreted.
- The format is documented with one example per kind.
- Code lives outside the legacy `cli.py` monolith.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
