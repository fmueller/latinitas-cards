---
id: T-055-reconcile-managed-fields-and-destination-tag
title: Reconcile managed fields and destination tag ownership
status: todo
priority: high
spec_ref: specs/v0.2.0.md#destination-tag-ownership
dependencies:
    - T-054-implement-bound-destination-snapshots-and-observed
updated_at: "2026-10-04T23:05:46Z"
---

# T-055-reconcile-managed-fields-and-destination-tag Reconcile managed fields and destination tag ownership

## Description

Build the deterministic three-way field/tag reconciliation used by managed plans, preserving destination-only user data.

## Acceptance

- Managed fields change safely only when destination equals baseline; destination equal to proposal is convergent/no-write. Divergence from both is a conflict, including destination-only edits regeneration would undo.
- Resolve explicitly by keep destination, accept proposal, or reviewed replacement; record decisions and resulting managed values. Personal Notes and other user-owned fields are never writable or offered as overridable conflicts.
- Preserve destination-only tags, separate source/configured/lifecycle contributions, and retain tags with overlapping origins when only one contribution disappears.
- Show managed-tag removal decisions; a deleted but still-required managed tag is a conflict. Explicit keep-as-user-owned overrides persist across later plans. Unknown initial ownership requires review.
- Managed-looking destination additions are not automatically tool-owned. Reserved lifecycle collisions require resolution and never authorize suspension. Do not infer invisible user intent.
- Test exact final tag sets for additions/removals, overlap, user deletion/addition, ownership overrides, collisions, unknown baseline, and no-op regeneration; sibling notes share the same reconciled tags.

## Verification

Use asymmetric three-way fixtures and red/green tests. Run the mandatory ruff, mypy, pytest -v chain.
