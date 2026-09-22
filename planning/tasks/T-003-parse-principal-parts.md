---
id: T-003-parse-principal-parts
title: Parse principal parts into semantic roles
status: todo
priority: high
spec_ref: specs/v0.1.0.md#principal-part-card-generation
dependencies:
    - T-001-define-deck-profile-contract
    - T-010-add-canonical-deck-source-adapters
updated_at: "2026-09-22T16:09:47Z"
---

# T-003-parse-principal-parts Parse principal parts into semantic roles

## Description

Create the domain parser that turns a confirmed lexical entry into named principal-part
roles or a structured unsupported/ambiguous result. Keep parsing and normalization separate
from card wording and rendering so both initial recipes share one verified interpretation.

## Acceptance

- A documented support matrix names accepted principal-part layouts, separators, omitted
  forms, and semantic roles.
- Parsing returns a typed successful result or a structured incomplete, unsupported, or
  ambiguous result that identifies the failed assumption.
- Normalization rules distinguish comparison-only normalization from preserved display
  content and identity-relevant semantic roles.
- The parser never guesses when multiple semantic-role interpretations remain.
- Local fixtures cover regular, deponent or otherwise supported exceptional entries,
  varying separators, omitted forms, malformed data, and ambiguity.
- Implement with red/green TDD in focused modules rather than expanding `cli.py`.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- Principal-part completion and recognition recipes are implemented by a dependent task.
