---
id: T-022-render-authored-note-kinds-into-dedicated-note
title: Render authored note kinds into dedicated note types
status: todo
priority: high
spec_ref: specs/v0.1.1.md#authored-note-kinds
dependencies:
    - T-020-parse-and-validate-authored-note-import-files
    - T-021-derive-stable-identities-for-authored-notes
updated_at: "2026-09-23T21:23:04Z"
---

# T-022-render-authored-note-kinds-into-dedicated-note Render authored note kinds into dedicated note types

## Description

Map `vocab`, `form`, and `qa` items onto one dedicated, stable note type each, sharing
the v0.1.0 note contract, with the default card directions from the spec.

## Acceptance

- Each kind has a stable note type name and field order with `LatinitasID` first,
  managed content fields, provenance, generation metadata, and a personal-notes field.
- `vocab` renders Latin to meaning; `form` renders the form (with context when present) to
  base form, analysis, and translation; `qa` renders question to answer.
- Contextually distinct occurrences of the same text form use distinct explicit keys.
  Tests show these yield separate items even with the same form text, while a citation
  edit alone preserves an established key's identity.
- Reference field definitions and front/back templates document creation of each stable
  note type with one card per authored item, including the user-owned Personal Notes field.
- Rendering is covered by tests for each kind, including optional fields left empty.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
