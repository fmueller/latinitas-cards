---
id: T-025-ship-the-authored-note-extraction-agent-skill
title: Ship the authored note extraction agent skill
status: todo
priority: medium
spec_ref: specs/v0.1.1.md#authored-note-extraction-skill
dependencies:
    - T-020-parse-and-validate-authored-note-import-files
    - T-023-preview-and-select-authored-notes-before-export
updated_at: "2026-09-23T21:23:04Z"
---

# T-025-ship-the-authored-note-extraction-agent-skill Ship the authored note extraction agent skill

## Description

Add a repository agent skill, installed for both `.claude/skills/` and
`.agents/skills/`, that extracts authored items from loosely structured Markdown study
notes into the import format and then runs validation and preview.

## Acceptance

- The skill documents key conventions per kind and requires explicit, reviewed keys for
  `qa` items.
- It covers varying heading and table column names, bullet-list vocabulary, and answers
  given inline or in a separate solutions section.
- Re-extraction into an existing import file preserves existing keys and `skip` decisions
  and reports new, changed, and missing items.
- It runs validation and preview before reporting, and never writes to Anki directly.
- The skill is corpus-agnostic; its examples use synthetic notes from more than one text
  and contain no private study material.
- A synthetic Markdown fixture and its expected import file are checked in and validate
  cleanly.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
