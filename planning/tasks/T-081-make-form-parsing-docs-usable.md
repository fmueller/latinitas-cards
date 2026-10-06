---
id: T-081-make-form-parsing-docs-usable
title: Make form-parsing docs usable without reading tests
status: todo
priority: low
spec_ref: specs/v0.2.0.md#calibrated-form-parsing
dependencies: []
updated_at: "2026-10-06T20:17:05Z"
---

# T-081-make-form-parsing-docs-usable Make form-parsing docs usable without reading tests

## Description

Found by the T-069 new-user walkthrough. docs/contextual-form-parsing.md has no complete
accepted-decision example, refers to fixtures that live only in tests, mentions internal task
IDs and Python callers, and points to `REFERENCE_CARD_CSS` in source instead of the reference
note type doc. Not release-blocking.

## Acceptance

- A sanitized example cases.json produces at least one exportable exercise.
- User docs link the reference CSS section and omit internal task IDs and Python-only instructions.

## Verification Notes

- Pending.

## Implementation Notes


