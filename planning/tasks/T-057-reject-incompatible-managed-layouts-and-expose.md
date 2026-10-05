---
id: T-057-reject-incompatible-managed-layouts-and-expose
title: Reject incompatible managed layouts and expose explicit fresh starts
status: completed
priority: high
spec_ref: specs/v0.2.0.md#pre-release-migration-boundary
dependencies:
    - T-054-implement-bound-destination-snapshots-and-observed
updated_at: "2026-10-05T03:33:21Z"
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

## Implementation Notes

- Extended T-034 policy in `legacy_transition.py`; bound managed snapshots reject
  historical IDs and incompatible schema/template layouts with actionable choices.
  Explicit fresh-start plans validate separate collection/type bindings, backup,
  disposable-data and new-schedule approvals without normalizing old inventory.
  Original data stays read-only; history/personal-note cases retain/defer.
- Strict RED: four layout regressions failed (legacy identity accepted; other
  layouts lacked choices), and four fresh-start tests failed for the absent API.
  GREEN: 95 focused tests passed. Initial full chain passed with 767 tests.
- Dedicated code-simplifier Task loaded its skill, made no changes, and passed
  74 focused tests. Independent parallel General, Security and Python lanes loaded
  code-reviewer and mapped companions. Database lane omitted: no SQL, native
  collection writes or migrations; input-binding/data-loss risk covered by Security.
- Candidate validation rejected F-057-1 and ST-1 as worded: collection-local IDs
  and display names are distinct namespaces; duplicate underlying gap covered by
  T057-PY-1. Validated finding verbatim: "Require an explicit approved binding
  between the new destination’s collection-local `schema.note_type_id` and the
  profile’s selected generated note type; the current checks can approve a fresh
  start whose destination is bound to a different note type."
- T057-PY-1 fixed with an explicit approved native type ID, separately checked
  against the bound ID while the approved display name matches the profile.
  Behavioral RED: missing/wrong IDs both DID NOT RAISE; GREEN: 97 focused tests.
  Fresh disposition-verification Task confirmed RESOLVED and "New task-relevant
  findings: None." No deferred findings or new follow-up tasks.
- Final exact chain: `uv run ruff check`; `uv run mypy`; `uv run pytest -v`:
  clean Ruff, no issues in 78 source files, 769 passed. Manual old-identity check
  confirmed actionable rejection. No native isolation or preservation claim;
  native update validation remains the existing later gate. Structural CSV
  effects and scheduling-preserving consolidation remain unsupported.
- 2026-10-05T03:33:21Z: verification pass
- 2026-10-05T03:33:21Z: Reviewed offline legacy layout rejection and explicit separate-destination fresh-start plans. Native isolation proof remains later update gate; no native claim.
