---
id: T-061-verify-managed-csv-import-safety-and-recovery-in
title: Verify managed CSV import safety and recovery in Anki
status: todo
priority: high
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies:
    - T-059-emit-approved-managed-csv-updates-and-reconcile
updated_at: "2026-10-04T23:05:46Z"
---

# T-061-verify-managed-csv-import-safety-and-recovery-in Verify managed CSV import safety and recovery in Anki

## Description

Exercise the supported native import/update path on a sanitized multi-card representative deck. This is an evidence gate, not a unit-test proxy for destination safety.

## Acceptance

- Record transport, Anki client/version, fixture setup, import settings, backup, and before/after offline destination observations.
- Verify content/tag updates and no-op reapplication produce no duplicate notes/cards, unchanged Personal Notes and destination-only tags, stable sibling bindings, and preserved scheduling/review histories for each card, including user-suspended cards.
- Separately test tag-only addition, removal, and removal of the final tag with every managed field unchanged; compare exact observed tag sets. Skipped intended changes remain pending or unsupported. Do not force an update by modifying unrelated content or metadata.
- Compare complete card and review-log tables and enumerate only expected note/content/tag changes; inspect sibling relationships and configured burying without implementing the scheduler.
- Exercise stale-plan rejection, changed GUI-handoff state, partial import reconciliation, backup/recovery, and retry, including import success before local recording fails, mixed field/tag outcomes, and backup restoration after recorded success. Verify approved subsets cannot overwrite kept fields or apply unapproved operations.
- Verify unsupported additions/retirement/reactivation/consolidation are refused and content-only plans cannot indirectly add or remove cards. Include a formerly absent slot becoming eligible during a sibling update and an enabled slot with a missing destination card; compare the actual card set.
- Verify separate-destination fresh starts leave original notes/cards/user data unchanged and disclose new scheduling.
- Run existing-recipe checks when the direct prerequisite is complete; form-parsing-exercises owns affected retests after its later changes. Bind evidence to tested content/setup and repeat affected checks for release-candidate changes.
- Publish exact supported capabilities and limitations. A failed or unavailable native run remains an open gate, not completed safety evidence. No actual user collection is required.

## Verification

Record reproducible sanitized observations and decisive table comparisons. Run the mandatory validation chain if fixes change code.
