---
id: T-002-stable-generated-note-identity
title: Define stable generated-note identity
status: todo
priority: high
spec_ref: specs/v0.1.0.md#stable-generated-note-identity
dependencies:
    - T-001-assisted-deck-profiles
updated_at: "2026-09-21T22:35:11Z"
---

# T-002-stable-generated-note-identity Define stable generated-note identity

## Description

Introduce the identity and ownership contracts required before users accumulate review
history on generated cards. `LatinitasID` must be stable across content, formatting, tag,
and software-version changes, and must not depend on local Anki note/card IDs. Preserve the
source note GUID for APKG/COLPKG inputs and an explicit or manifest-assigned source identity
for CSV inputs.

## Acceptance

- A documented identity function uses immutable source identity, recipe identity, and a
  semantic exercise key.
- Tests prove mutable prompt, answer, gloss, HTML, tag, and software-version changes retain
  identity while semantically distinct exercises receive distinct IDs.
- APKG/COLPKG readers retain source note GUIDs; CSV profiles require or persist stable
  source IDs.
- The generated-note contract separates immutable identity, managed content, provenance,
  recipe metadata, and a user-owned personal-notes field.
- `LatinitasID` is suitable as the first Anki text-import field; future deterministic Anki
  GUID derivation is specified from the same logical identity.
- Relationships use source/Latinitas identities and never local Anki card IDs.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- Replace, rather than reuse, the current GUID seed that includes a newly allocated note ID.
