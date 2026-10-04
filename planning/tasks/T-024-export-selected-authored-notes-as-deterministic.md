---
id: T-024-export-selected-authored-notes-as-deterministic
title: Export selected authored notes as deterministic CSV
status: todo
priority: high
spec_ref: specs/v0.1.1.md#authored-note-selection-and-preview
dependencies:
    - T-023-preview-and-select-authored-notes-before-export
    - T-004-preview-and-deterministic-csv-export
updated_at: "2026-09-23T21:23:04Z"
---

# T-024-export-selected-authored-notes-as-deterministic Export selected authored notes as deterministic CSV

## Description

Write the selected authored notes through the v0.1.0 CSV export boundary so repeated
imports update the same logical notes.

## Acceptance

- Export writes UTF-8 Anki-import CSV per note type with stable field order, target deck,
  provenance tags, and import metadata or precise instructions.
- Same import file, namespace, and filters produce byte-stable output.
- Tests prove edited content keeps identity, `skip` items are absent, and Personal Notes
  is omitted from CSV columns rather than written empty; check headers and row widths for
  every kind. Import instructions leave Personal Notes unmapped.
- Invalid input anywhere in the file, including skipped or filtered-out rows and conflicting
  duplicates, exits nonzero before any output writes. CLI tests verify existing output
  stays unchanged and no new output is created. Empty selection reports zero without writes.
- Documentation covers first import, re-import after edits, and combined selection filters,
  with explicit namespace configuration and no corpus resources or principal-part mappings.
- Record a supported-Anki first-import/re-import check for each kind: edited content updates
  the same note, nonempty Personal Notes survives, and the note yields one intended card.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
