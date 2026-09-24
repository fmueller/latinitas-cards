---
id: T-012-build-assisted-profile-setup
title: Build assisted profile setup
status: completed
priority: high
spec_ref: specs/v0.1.0.md#assisted-deck-profiles
dependencies:
    - T-001-define-deck-profile-contract
    - T-010-add-canonical-deck-source-adapters
    - T-003-parse-principal-parts
    - T-011-validate-the-representative-deck-and-german
updated_at: "2026-09-24T20:10:01Z"
---

# T-012-build-assisted-profile-setup Build assisted profile setup

## Description

Build the interactive CLI workflow that inspects canonical source records, proposes
mappings and compatible recipes, shows representative examples, requires confirmation, and
saves a reusable profile.

## Acceptance

- Setup proposes note type, lexical field, principal-part field, optional German gloss,
  separators, named role ordering, generated note type, target deck, tags, and compatible
  recipes.
- Representative source values and uncertainty are shown before confirmation; no profile
  is saved without explicit confirmation.
- Rejected or corrected proposals are reflected in the saved profile without modifying the
  source deck.
- Explicit CLI overrides follow the profile contract and the effective configuration is
  reported in human- and machine-readable forms.
- Reusing a confirmed profile is deterministic and does not repeat setup unless requested
  or invalidated.
- CLI controls and diagnostics are English, while generated-content language uses explicit
  tags.
- Tests exercise confirmed, corrected, cancelled, unsupported-schema, and non-interactive
  profile-reuse flows across CSV and Anki package fixtures.
- Command code remains thin and outside the legacy `cli.py` monolith.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- 2026-09-24T20:09:51Z: verification pass
