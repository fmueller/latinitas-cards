---
id: T-001-define-deck-profile-contract
title: Define the deck profile contract
status: completed
priority: high
spec_ref: specs/v0.1.0.md#assisted-deck-profiles
dependencies: []
updated_at: "2026-09-23T19:38:06Z"
---

# T-001-define-deck-profile-contract Define the deck profile contract

## Description

Define the versioned, human-readable profile and effective-configuration domain model used
by source inspection, assisted setup, identity, and generation. Keep serialization,
validation, defaults, and explicit CLI overrides independent from interactive prompting and
from individual CSV/APKG/COLPKG adapters.

## Acceptance

- The versioned profile records source identity, note type, fields, separators, named
  principal-part roles, language tag, generated-note type, target deck, tags, and selected
  recipes.
- Profile validation reports missing, incompatible, and contradictory values without
  depending on deck-specific field names.
- Loading the same profile and explicit overrides produces the same effective configuration;
  that configuration has human- and machine-readable representations.
- Unsupported profile schema versions fail clearly, and profiles contain no credentials.
- Domain and serialization tests cover round trips, override precedence, defaults, language
  tags, selected recipes, and schema-version failures.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- Keep profile/domain code outside the legacy `cli.py` monolith. Source adapters and the
  assisted confirmation workflow are separate tasks.
- 2026-09-23T19:37:05Z: verification pass
- 2026-09-23T19:38:06Z: verification pass
