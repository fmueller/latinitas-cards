---
id: T-059-emit-approved-managed-csv-updates-and-reconcile
title: Emit approved managed CSV updates and reconcile import results
status: todo
priority: high
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies:
    - T-058-expose-deterministic-managed-plans-and-operation
updated_at: "2026-10-04T23:05:46Z"
---

# T-059-emit-approved-managed-csv-updates-and-reconcile Emit approved managed CSV updates and reconcile import results

## Description

Implement the supported content/tag CSV artifact and offline reconciliation workflow. Application includes the documented user-mediated native import; artifact generation alone is not successful application.

## Acceptance

- Revalidate bound snapshot/baseline/schema/plan preconditions before emission; changed state requires replanning and renewed approval. Reject unsupported structural effects and unsafe content subsets.
- Emit only approved managed content and the exact reconciled tag set for a compatible existing note type. Omit Personal Notes and user-owned fields from importable writes. Verify the approved-subset fixture against actual emitted columns/import mapping; any required unchanged column uses its resolved destination value.
- Require a recoverable destination backup and report approved, emitted, pending, observed, failed, and unresolved operations without confusing these states.
- Document fresh snapshot and no intervening edits through GUI import, then acquire an observed result snapshot. If that condition cannot be established, withhold preservation claims and require reconciliation/replanning.
- Validate observed results against approved operations before baseline advancement. Partial/mismatched imports advance only confirmed reconciled operations; retries inspect observed destination state before emitting pending operations. A skipped tag-only update remains pending or unsupported; never change unrelated content or metadata to force it.
- Tests cover stale approval, emission failure, no import, wrong destination, partial import, destination edits during the handoff, recovery, and idempotent reconciliation/retry. Preserve the prior-export transaction regressions.
- Cover successful import followed by failed/interrupted result or baseline persistence, mixed field/tag outcomes within one note, and destination backup restoration after successful operations were recorded. Reacquire evidence and reconcile or invalidate inconsistent baselines before retry; export rollback alone is not applied-baseline crash-recovery evidence.

## Verification

Use red/green tests and the mandatory ruff, mypy, pytest -v chain. Do not claim native safety before the native update gate passes.
