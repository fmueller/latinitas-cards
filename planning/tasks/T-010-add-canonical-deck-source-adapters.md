---
id: T-010-add-canonical-deck-source-adapters
title: Add canonical deck source adapters
status: completed
priority: high
spec_ref: specs/v0.1.0.md#assisted-deck-profiles
dependencies: []
updated_at: "2026-09-24T00:48:38Z"
---

# T-010-add-canonical-deck-source-adapters Add canonical deck source adapters

## Description

Expose CSV, APKG, and COLPKG inputs as immutable canonical source records for profile setup
and generation. Preserve Anki note GUIDs and named fields without making deck-specific
assumptions or modifying the source.

## Acceptance

- CSV, APKG, and COLPKG adapters return one typed canonical record shape with source kind,
  available note type, named fields, provenance, and source identity when present.
- APKG/COLPKG records retain the source note GUID rather than only local numeric note IDs.
- CSV records retain an explicit stable source-ID column when configured and otherwise
  expose the data needed for manifest reconciliation.
- Inspection supports multiple note types and arbitrary field names, with deterministic
  ordering.
- Errors identify the source and failed structural assumption without unnecessarily
  exposing deck contents.
- Tests cover legacy and modern Anki schemas, CSV encodings and field names, malformed
  packages, and proof that source files remain unchanged.
- Code lives outside the legacy `cli.py` monolith.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- Reuse validated archive/database mechanics where useful, but do not reuse experimental
  APKG mutation as an output path.
- 2026-09-24T00:48:26Z: verification pass
