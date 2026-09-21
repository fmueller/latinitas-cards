---
id: T-001-assisted-deck-profiles
title: Add assisted profiles for existing decks
status: todo
priority: high
spec_ref: specs/v0.1.0.md#assisted-deck-profiles
dependencies: []
updated_at: "2026-09-21T22:35:11Z"
---

# T-001-assisted-deck-profiles Add assisted profiles for existing decks

## Description

Build the v0.1.0 entry point for arbitrary existing CSV, APKG, and COLPKG decks. Inspection
must propose note-type and field mappings, separators, principal-part roles, output deck and
note type, tags, and the German gloss language. The user confirms representative examples,
and LatinitasCards saves the result as a reusable, human-readable profile instead of
hard-coding one deck layout.

## Acceptance

- A sanitized fixture represents a realistic existing deck with lexical, principal-part,
  and German-gloss fields.
- Assisted setup proposes mappings with representative values and requires confirmation
  before saving them.
- The versioned profile records source identity, note type, fields, separators, named
  principal-part roles, language tag, generated-note type, target deck, tags, and selected
  recipes.
- CSV inputs and APKG/COLPKG inputs are covered without assuming fixed field names.
- Reusing a profile is deterministic; CLI overrides are reflected in human- and
  machine-readable effective configuration.
- Unsupported profile schema versions fail clearly, and profiles contain no credentials.
- The source input is never modified.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- Keep profile/domain code outside the legacy `cli.py` monolith.
