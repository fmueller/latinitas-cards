---
id: T-004-preview-and-deterministic-csv-export
title: Preview and export deterministic Anki CSV
status: todo
priority: high
spec_ref: specs/v0.1.0.md#preview-and-deterministic-csv-export
dependencies:
    - T-001-assisted-deck-profiles
    - T-002-stable-generated-note-identity
    - T-003-principal-part-card-generation
updated_at: "2026-09-21T22:35:11Z"
---

# T-004-preview-and-deterministic-csv-export Preview and export deterministic Anki CSV

## Description

Add a safe preview and file-output boundary for generated cards. Users must see
representative cards plus generated/skipped/ambiguous counts before writing deterministic,
UTF-8 Anki-import CSV. Repeated exports update the same logical notes through the first-field
`LatinitasID` contract and never mutate the source deck.

## Acceptance

- Preview renders representative prompt/answer/provenance content and reports generated,
  skipped, and ambiguous entries with structured reasons.
- CSV output places `LatinitasID` first and uses stable field order, dedicated note type,
  configurable deck/subdeck, provenance tags, and import metadata or precise instructions.
- Same input and effective profile produce byte-stable output or a documented set of
  intentionally variable metadata excluded from deterministic comparison.
- Tests prove wording, gloss, HTML, and tag updates retain identity and do not add duplicate
  logical rows.
- Source CSV/APKG/COLPKG fixtures are unchanged after preview and export.
- Human-readable output has a machine-readable equivalent suitable for future agents.
- Documentation demonstrates first import and repeat import settings in supported Anki.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- This task does not add live Anki updates or reuse experimental APKG mutation as the new
  output path.
