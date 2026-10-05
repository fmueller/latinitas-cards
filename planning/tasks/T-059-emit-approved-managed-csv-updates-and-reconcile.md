---
id: T-059-emit-approved-managed-csv-updates-and-reconcile
title: Emit approved managed CSV updates and reconcile import results
status: completed
priority: high
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies:
    - T-058-expose-deterministic-managed-plans-and-operation
updated_at: "2026-10-05T04:58:48Z"
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

## Implementation and workflow-v3 evidence

- Step 1: fetched origin/main; local HEAD and origin/main both matched accepted
  T-058 base `7c19ab0f494a36bca63f8c19b0fbc05817191e60`, with ancestry confirmed.
  Before writes, Taskrail validate returned state valid; next JSON selected
  exactly this task, off_spec false. Status confirmed active v0.2.0, no owner,
  T-058 completed. Read full task/spec, T-048 acceptance and transport contract,
  managed plan approval, reconciliation/lifecycle and atomic state APIs.
  Started exactly one task; no other task or thread was started.
- Step 2: domain RED failed collection with missing managed_application module;
  GREEN 4 tests passed. Mapping/CLI RED: wrong mapping did not raise and emit
  command absent; GREEN 6 passed. Missing-output-directory RED raised unreported
  FileNotFoundError; GREEN 9 passed. Strengthened mixed-result test was checked
  by deliberately removing exact-tag observation matching: failed with
  unreviewed tag ownership; restored the gate and all focused tests passed.
- Changes: managed_application emits only reverified selected CSV rows, using
  the existing serializer and approved effective-profile import metadata. It
  persists pending targets/backup before no-clobber atomic publication and
  reports stages separately. Observation reuses indivisible per-note exact
  field/tag/preservation matching and atomic anchor/receipt persistence.
  CLI adds emit/observe/reconcile outside cli.py. Kept destination values,
  manual tags, unselected notes, omitted Personal Notes and actual CSV mapping
  are tested. Initial reviewed field/tag differences may be retained without
  pretending prior-export checkpoints are baselines; preservation differences
  and inconsistent successful receipts still require reconciliation.
- Step 3: exact mandatory chain initially exposed a mypy import-export test
  access and outdated CLI command-set assertion; fixed and restarted the full
  chain. Initial green: ruff passed, mypy 84 files, pytest 820 passed. Prior
  export transaction regressions remain in the full suite unchanged.
- Step 4: dedicated Task loaded code-simplifier; inspected task-local diff and
  new files, made no changes. It found no clearly safe simplification and ran
  the focused plan/state/application tests: 67 passed. No suggestions rejected.
- Step 5: separate parallel read-only Tasks loaded code-reviewer in General,
  Security and Database lanes. General loaded ECC code-reviewer/common rules;
  Security loaded security-reviewer/security-review/common security; Database
  loaded database-reviewer/postgres-patterns/database-migrations plus relevant
  file-persistence guidance. General covers acceptance/tests, Security covers
  user-data/backup trust boundaries, Database covers atomic anchors/receipts,
  interruption, partial outcomes and restoration. Two specialists fit budget.
  Framework lanes omitted: no framework change. Initial Python specialist
  omitted in favor of material data-loss/persistence risks; final verification
  additionally loaded Python reviewer/python-patterns. No ML/domain analysis
  or native UI claim changed. General: "No concrete task-relevant findings."
  Fresh candidate-validation Task loaded code-reviewer and validated SEC-1
  and DB-1; no candidates rejected/deduplicated. SEC-1's original assertion
  that save_state shared the emission exception handler was inaccurate;
  validation confirmed only the generic-error/uncertain-artifact reporting gap.
- Step 6, SEC-1 verbatim: "Do not classify a journal-persistence error after
  successful CSV publication as an emission failure; preserve the
  pending/unknown outcome and report that the artifact may already exist."
  Fixed the validated reporting gap: post-publication persistence uncertainty
  returns structured emission unknown, no failed-success claim, explicit
  reload/inspect-before-retry instruction, and nonzero CLI status. Disk may
  contain pending or emitted state, so actual evidence must be reacquired.
- Step 6, DB-1 verbatim: "Reject a backup path that resolves to the
  applied-baseline journal; otherwise the journal itself can be recorded as
  the recoverable destination backup." Fixed samefile rejection before writes,
  covering direct, symlink and hardlink aliases. Disposition RED: 4 failures
  (three aliases did not raise, post-publication save raised generic OSError);
  GREEN 13 application tests passed. No finding deferred.
- Additional direct inspection found selected reconciliation decisions were
  not retained in observed target ownership. RED failed KeyError decisions;
  fixed merging prior and only selected field/tag decisions into targets.
  Added confirmed-import/restoration/review/replan/new-approval retry and
  mixed-results/review/replan/remaining-tags-only cases. Historical receipts
  survive restoration review; no dummy content edit forces tag-only Update.
- Step 7: fresh read-only disposition-verification Task loaded code-reviewer
  plus General/Python guidance; "SEC-1 — RESOLVED", "DB-1 — RESOLVED", and
  "No concrete task-relevant findings." One review/fix/recheck cycle. It ran
  diff check, ruff, mypy (84 files), pytest -q (825 passed). Parent exact final
  chain: ruff passed; mypy 84 files; pytest -v 825 passed. Focused tests:
  72 passed. No checks suppressed and no unresolved review findings.
- Executable manual CLI smoke in a temporary synthetic fixture: emitted=1,
  pending=1, observed=0; equal result without interval confirmation unresolved=1;
  confirmed interval observed=1. Temporary data cleaned. This is offline CLI
  evidence, not native GUI/card scheduling proof. Docs specify fresh capture,
  recoverable full backup/operator restoration attestation, GUI matching/mapping,
  no edits through result capture, persistence uncertainty and observed retry.

### Remaining scope and evidence

Native matching, full scheduling/history preservation and unchanged-field
tag-only semantics remain the existing T-061 gate, not proven here. Snapshot
and backup recoverability assertions are operator reviewed, not inferred from
file names or hashes; file output cannot enforce the GUI no-edit interval.
Unknown interval withholds claims and requires reconciliation/replanning.
Structural/card additions, retirement/reactivation, slot prompt/answer/guard
changes, APKG and incompatible CSS migration remain unsupported; CSS setup
is manual. Source-only v0.1 export remains limited and unchanged. No new
follow-up needed: existing pinned tasks own the remaining release gates.

## Implementation Notes

- 2026-10-05T04:58:48Z: verification pass
- 2026-10-05T04:58:48Z: Workflow-v3 steps 1-7 passed before verify/complete. Reviewed selected CSV handoff, backup alias protection, explicit uncertain emission reporting, atomic observed per-note receipts, recovery/replan/retry and decision provenance. Both review findings resolved, no deferrals. Native client matching/tag-only/preservation remains T-061; source-only v0.1 path unchanged.
