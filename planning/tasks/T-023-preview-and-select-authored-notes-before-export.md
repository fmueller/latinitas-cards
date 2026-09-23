---
id: T-023-preview-and-select-authored-notes-before-export
title: Preview and select authored notes before export
status: todo
priority: high
spec_ref: specs/v0.1.1.md#authored-note-selection-and-preview
dependencies:
    - T-022-render-authored-note-kinds-into-dedicated-note
updated_at: "2026-09-23T21:23:04Z"
---

# T-023-preview-and-select-authored-notes-before-export Preview and select authored notes before export

## Description

Add side-effect-free validation and preview for an import file, plus selection filters
by kind, source section, source reference, and tag.

## Acceptance

- Preview reports counts by kind, source section, source reference, and status; merged
  duplicates; invalid or conflicting items with reasons; and representative cards.
- Filters narrow the selection without modifying the import file; the effective selection
  is reported.
- `skip` items are counted but never selected for export.
- Preview consumes an internal typed result separate from terminal rendering.
- Tests prove preview writes no files and leaves the import file unchanged.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
