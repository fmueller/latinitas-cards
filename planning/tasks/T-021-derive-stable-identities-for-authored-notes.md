---
id: T-021-derive-stable-identities-for-authored-notes
title: Derive stable identities for authored notes
status: todo
priority: high
spec_ref: specs/v0.1.1.md#authored-note-identity
dependencies:
    - T-020-parse-and-validate-authored-note-import-files
    - T-002-stable-generated-note-identity
updated_at: "2026-09-23T21:23:04Z"
---

# T-021-derive-stable-identities-for-authored-notes Derive stable identities for authored notes

## Description

Derive each authored item's `LatinitasID` from collection namespace, kind, and normalized
key, reusing the v0.1.0 identity contract. Merge compatible duplicates and reject
conflicting ones.

## Acceptance

- Identity depends only on namespace, kind, and normalized key; tests prove content,
  wording, tags, status, provenance text, and line order do not change it.
- Key normalization is deterministic and documented.
- Duplicate items with identical or compatible content merge and are reported; conflicting
  duplicates fail and name both lines and differing fields. Required content, language,
  document, and section must agree; absent optional content or references may be filled,
  but different nonempty values conflict. Tags are unioned and sorted; `skip` dominates.
- Tests reverse duplicate order to prove identical merged results, including mixed status,
  tag union, and optional-field completion. Normalized-key collisions with conflicting
  answers, language, or provenance fail rather than selecting one row's values.
- Different namespaces or kinds with the same key yield distinct identities.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
