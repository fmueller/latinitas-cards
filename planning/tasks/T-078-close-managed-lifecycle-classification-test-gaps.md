---
id: T-078-close-managed-lifecycle-classification-test-gaps
title: Close managed lifecycle and classification test gaps
status: todo
priority: low
spec_ref: specs/v0.2.0.md#note-and-card-lifecycle-contract
dependencies: []
updated_at: "2026-10-06T20:17:05Z"
---

# T-078-close-managed-lifecycle-classification-test-gaps Close managed lifecycle and classification test gaps

## Description

Found by the T-069 review. Missing tests: conflicting suspension ownership, retiring an
already user-suspended card, the changed tool-suspension branch, and a card-level `reactivate`
operation refused at approve. A managed note deleted in the destination is classified `create`
rather than a conflict, and a `retire` classification hides an underlying conflict.
Not release-blocking (lifecycle operations are unsupported at apply).

## Acceptance

- Tests cover the four listed lifecycle branches.
- A previously anchored note missing from a complete destination is a reviewed conflict, not `create`; retire entries keep their conflict reasons visible.

## Verification Notes

- Pending.

## Implementation Notes


