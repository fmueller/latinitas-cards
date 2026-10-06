---
id: T-071-record-explicit-decisions-for-unselected-tag-conflicts
title: Record explicit tag decisions for unselected managed tag conflicts
status: todo
priority: high
spec_ref: specs/v0.2.0.md#destination-tag-ownership
dependencies: []
updated_at: "2026-10-06T20:16:22Z"
---

# T-071-record-explicit-decisions-for-unselected-tag-conflicts Record explicit tag decisions for unselected managed tag conflicts

## Description

Found by the T-069 adversarial review. When a generated tag was deleted in the destination
(a tag conflict) and only a field operation was approved, emit's fallback ownership recorded
the tag as `suppressed_tags` with no decision, and the next plan reported the note as
unchanged. specs/v0.2.0.md#destination-tag-ownership requires that such a conflict be
resolved explicitly and the decision recorded. Outcome is not data loss but fabricates
provenance. Release-blocking for T-068.

## Acceptance

- Observing a plan whose tags operation was not selected never adds suppressed or kept tag ownership without a recorded reviewed decision.
- The user-deleted generated tag stays a visible conflict on the next plan until explicitly resolved.
- Regression test reproduces the delete-tag/approve-field-only/observe/replan sequence; mandatory chain passes.

## Verification Notes

- Pending.

## Implementation Notes


