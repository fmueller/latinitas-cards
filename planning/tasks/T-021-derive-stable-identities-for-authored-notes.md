---
id: T-021-derive-stable-identities-for-authored-notes
title: Derive stable identities for authored notes
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-identity
dependencies:
    - T-020-parse-and-validate-authored-note-import-files
    - T-002-stable-generated-note-identity
updated_at: "2026-10-04T10:05:27Z"
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

- Verification run 2026-10-04T10:05:27Z passed after the exact Ruff, mypy,
  and pytest chain: all lint checks passed, 57 typed files clean, 482 tests passed.
- Strict behavioral TDD: 18 failures against the initial API skeleton (normalization
  mismatch, absent duplicate result, missing conflict errors), then 18 passing tests.
- Dedicated code-simplifier loaded its skill and recommended no changes. Independent
  General, Python, and Security reviewers each reported: "No concrete task-relevant
  findings." Fresh candidate validation found zero candidates; fresh disposition
  verification found no unresolved or newly introduced task-relevant issues.
- Taskrail selector matched T-021 and the pinned v0.1.1 spec before start, verify,
  and complete. No follow-up task was needed; this cycle stops after T-021.

## Implementation Notes

- Authored identity and whole-file reconciliation live in authored_identity.py,
  reusing the v0.1.0 derive_latinitas_id contract. Keys use NFC and collapsed
  whitespace while preserving case; namespaces and content remain verbatim.
- Compatible duplicates complete absent optional values, sort tag unions, and
  honor skip dominance. Conflicts retain actual contributor lines and field names.
- CLI, selection, and export remain outside this task's scope.
- 2026-10-04T10:05:27Z: verification pass
