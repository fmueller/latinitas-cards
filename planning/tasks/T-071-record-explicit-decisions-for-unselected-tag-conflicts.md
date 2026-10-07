---
id: T-071-record-explicit-decisions-for-unselected-tag-conflicts
title: Record explicit tag decisions for unselected managed tag conflicts
status: completed
priority: high
spec_ref: specs/v0.2.0.md#destination-tag-ownership
dependencies: []
updated_at: "2026-10-07T20:18:30Z"
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

- 2026-10-07T20:18:30Z: verification pass
- 2026-10-07T20:18:30Z: Keep prior reviewed tag ownership/baseline for unselected tag operations; journal baseline_tags and anchor observed_tags separate provenance from observation, paralleling T-070 field semantics. RED three delete-tag/field-only variants exposed fabricated suppression; GREEN 91 focused tests, final ruff/mypy/pytest 843 passed. Dedicated simplifier: no edits. General, persistence, Python, Security reviews, fresh candidate validation and disposition verification: no concrete task-relevant findings; no dispositions or follow-ups. Exact destination tags and selected fields preserved; deleted generated tag remains conflict on replan and restored tags remain detectable. Verification run 2026-10-07; offline synthetic evidence only.
