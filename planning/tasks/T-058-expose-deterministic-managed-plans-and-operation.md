---
id: T-058-expose-deterministic-managed-plans-and-operation
title: Expose deterministic managed plans and operation-bound approval
status: completed
priority: high
spec_ref: specs/v0.2.0.md#managed-update-plans
dependencies:
    - T-055-reconcile-managed-fields-and-destination-tag
    - T-056-plan-identity-preserving-card-retirement-and
    - T-057-reject-incompatible-managed-layouts-and-expose
    - T-053-render-safe-principal-part-comparisons-with
updated_at: "2026-10-05T04:30:04Z"
---

# T-058-expose-deterministic-managed-plans-and-operation Expose deterministic managed plans and operation-bound approval

## Description

Compose the reconciled state into inspectable machine-readable plans and a CLI review/approval workflow. Keep new command/domain implementation out of cli.py.

## Acceptance

- Classify notes as create/update/unchanged/conflict/retire with reasons, field/tag differences, proposed values, per-card effects, blocked operations, and transport capabilities.
- Record destination/snapshot fingerprint, baseline version, effective profile, schema/template binding, and a deterministic plan identity. Stable inputs yield stable output and no-op regeneration produces no unintended writes.
- Approval is explicit and bound to the selected operations and resolved plan; generation, profile confirmation, knowledge review, or a lifecycle tag is not apply approval. Modified plans/resolutions, including presentation changes affecting the approved payload, require renewed approval.
- Include structural and lifecycle proposals for inspection while reporting their selected-transport application as unsupported. Permit separately approved compatible content-only subsets only when retention/eligibility makes them safe and actual guard fields cannot indirectly add or remove cards.
- Cover create, unchanged, safe update, convergent/divergent edits, missing baseline, wrong destination, duplicates, incomplete and stale snapshots in sanitized CLI/plan fixtures.
- Verify approval against CSV's actual write footprint using an asymmetric fixture: approve field A, keep destination field B, leave another operation unapproved, and retain destination-only tags. Required unchanged columns carry resolved destination values; emitted columns/import mapping must not apply an unapproved value.
- No personal fields can enter approved write payloads. Preserve existing source-only export as the explicitly limited v0.1 path.

## Verification

Use red/green domain and CLI tests and the mandatory ruff, mypy, pytest -v chain.

## Implementation and workflow evidence

- Pinned v0.2.0/all-open cycle: fetched origin/main and confirmed accepted
  ffd2107c7f7b3ad6f0a05588c62c7df9e5704731 is contained; checkout matched it.
  `taskrail validate` reported state valid; `next --json` selected this task.
- Added focused managed_plans domain and managed plan/approve command modules;
  cli.py changes are registration only. JSON plans bind snapshot, baseline,
  schema/template/style and effective profile, including presentation settings.
  Explicit selected-operation receipts contain exact import columns/rows and
  retain destination values for unselected fields/tags; omitted notes have no row.
- RED: focused collection failed with missing managed_plans module. GREEN:
  initial domain/CLI fixtures passed. A rehashed edited operation initially passed
  verification (DID NOT RAISE); recomputing derived plans made it fail closed.
  RED: asymmetric import-footprint test failed with missing import_columns;
  GREEN: selected Meaning, destination Lemma, omitted second-note operation and
  destination-only Tags are checked in actual planned import column positions.
- Initial full gate exposed the command-module test's assumption of only one
  Typer group; extending its managed-group assertions restored the full gate.
  Initial successful chain: ruff clean, mypy 82 files clean, pytest 806 passed.
- Dedicated Task simplifier loaded code-simplifier, inspected the actual change,
  made no edits, and passed 12 focused tests plus lint/type checks.
- Separate parallel read-only Tasks loaded code-reviewer in General, Security,
  and Python lanes with their mapped reviewer guidance. General had F1/F2;
  Security had SEC-1; Python: "No concrete task-relevant findings."
  Database omitted: no SQL/database or persistent application changes. UI
  omitted: no visual UI. A fresh candidate validator confirmed all three;
  none rejected or deduplicated. One review/fix cycle, no deferred findings.

### Verbatim validated findings and dispositions

- F1: "The managed plan drops the lifecycle planner’s proposed note-level
  retirement tag." Fixed: merge the lifecycle planner's proposed tag into
  inspected reconciliation contributions, with its tag operation unsupported.
- F2: "Tag reconciliation output omits the baseline tag set and tag-removal
  details from each note’s review plan." Fixed: expose baseline_tags and
  explicit additions/removals/removal_candidates while retaining user tags.
- SEC-1: "Managed approval can emit a Tags cell that does not preserve the
  destination tag set when an accepted destination tag contains whitespace."
  Fixed: existing tag-character validator rejects unrepresentable whitespace
  and control characters in destination, baseline and contribution evidence.
- RED: focused review regressions reported missing lifecycle-tag contribution,
  missing baseline_tags and tag validation DID NOT RAISE. GREEN: 17 passed,
  including whitespace/tab/ESC/DEL tag rejection. Fresh disposition-verification
  Task loaded code-reviewer, confirmed all three resolved with no new findings;
  focused managed/CLI suite: 64 passed; diff whitespace check passed.

### Final checks and boundary

- Exact final chain: `uv run ruff check` (All checks passed), `uv run mypy`
  (Success: no issues found in 82 source files), `uv run pytest -v` (811 passed).
- Actual installed CLI smoke test emitted an update plan and an explicit
  single-Meaning approval: one target/row, Personal Notes absent from columns,
  application/native/structural/lifecycle/slot capability flags false.
- No baseline or destination writes, no native preservation claim. Slot answer,
  prompt and guard changes remain unsupported by the conservative journal.
  Incompatible CSS requires manual setup. Source-only v0.1 export is unchanged.
  Existing T-059 owns actual managed CSV/reconciliation; T-061/T-062 own native
  gates. No additional follow-up was needed and no second task was started.

## Implementation Notes

- 2026-10-05T04:30:04Z: verification pass
