---
id: T-034-guard-legacy-note-model-transitions-with-explicit
title: Guard legacy note-model transitions with explicit safe options
status: todo
priority: high
spec_ref: specs/v0.1.0.md#pre-release-note-model-migration
dependencies:
    - T-030-generate-conditional-sibling-cards-with-stable
updated_at: "2026-09-27T08:56:30Z"
---

# T-034-guard-legacy-note-model-transitions-with-explicit Guard legacy note-model transitions with explicit safe options

## Description

Make incompatible pre-release model transitions explicit without an unproven
history-preserving merger or destructive collection changes.

## Acceptance

- Reject incompatible legacy profiles or require explicit re-confirmation; do not
  reinterpret old per-exercise identities as object identities. Inventory affected
  tests, examples and fixtures.
- Document and rehearse a backed-up fresh start into a new dedicated note type for
  explicitly approved disposable data, acknowledging new schedules.
- Keep old collections intact without specific cleanup approval. For valuable history
  or conflicting Personal Notes/tags, retain old collections and defer conversion;
  do not reconcile by last-write-wins.
- State that ordinary CSV structural consolidation has no demonstrated history guarantee.
  A future preservation path requires separate approval, destination-aware per-card
  mapping, annotation/tag reconciliation, rollback and native preservation evidence.
- Distinguish compatible new-model regeneration from migration; preserve surviving cards
  and history during compatible updates. No live collection mutation is required.

## Verification Notes

- Test legacy-schema rejection and no-write/review paths, including conflicting personal
  annotations. Rehearse the synthetic fresh-start checklist; record evidence when run.
- Run the mandatory ruff/mypy/pytest chain after code changes.

## Implementation Notes
- Complex migration solely for unreleased data is not required; no silent conversion.
