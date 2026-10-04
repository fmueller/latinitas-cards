---
id: T-057-reject-incompatible-managed-layouts-and-expose
title: Reject incompatible managed layouts and expose explicit fresh starts
status: todo
priority: high
spec_ref: specs/v0.2.0.md#pre-release-migration-boundary
dependencies:
    - T-054-implement-bound-destination-snapshots-and-observed
updated_at: "2026-10-04T23:05:46Z"
---

# T-057-reject-incompatible-managed-layouts-and-expose Reject incompatible managed layouts and expose explicit fresh starts

## Description

Extend completed T-034's legacy classification, transition policy, and regressions with destination-snapshot/schema/template checks and separate-destination enforcement. Do not reimplement that policy or introduce a general consolidation engine.

## Acceptance

- Detect incompatible identity/schema/template layouts, including historical per-exercise notes, and reject ordinary managed apply with actionable choices. Do not reinterpret old manifests or reuse a template slot for another task.
- Offer an explicitly selected fresh start in a separate destination after backup. State that new cards do not inherit schedules; leave original notes/cards, personal fields, tags, source structure, and review logs untouched.
- Scheduling-preserving consolidation remains unsupported and fails clearly; regeneration/CSV export is not migration. No automatic old-note deletion or retirement occurs.
- Tests cover old layouts, incompatible template bindings, explicit fresh-start selection, separate destination enforcement, and original-data preservation, retaining T-034's existing policy tests.

## Verification

Use red/green tests and the mandatory ruff, mypy, pytest -v chain. Verify fresh-start destination isolation in the native update gate.
