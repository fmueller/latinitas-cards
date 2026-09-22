---
id: T-011-validate-the-representative-deck-and-german
title: Validate the representative deck and German terminology
status: todo
priority: high
spec_ref: specs/v0.1.0.md#release-readiness
dependencies:
    - T-001-define-deck-profile-contract
    - T-010-add-canonical-deck-source-adapters
    - T-003-parse-principal-parts
updated_at: "2026-09-22T16:09:47Z"
---

# T-011-validate-the-representative-deck-and-german Validate the representative deck and German terminology

## Description

Validate provisional profile, source-record, and principal-part assumptions against a
sanitized representative deck or export and user-reviewed German wording before the
assisted setup and recipe deliverables are finalized.

## Acceptance

- A sanitized fixture represents realistic note types, arbitrary field names, separators,
  principal-part conventions, optional glosses, and stable source identity behavior without
  private deck content.
- The fixture is exercised through the canonical source adapters without product-specific
  hard-coding.
- Representative mappings, examples, recipe suggestions, German wording, and
  principal-part terminology are reviewed and recorded.
- Any mismatch with provisional schemas, parser rules, or terminology is resolved in the
  owning task before dependent work proceeds.
- The fixture's provenance and sanitization method are documented sufficiently for
  repository use.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- Synthetic fixtures may support earlier development, but they do not satisfy this
  acceptance gate.
