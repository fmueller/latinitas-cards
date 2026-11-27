---
id: T-246-record-approved-post-trial-contracts-and-task
title: Record approved post-trial contracts and task package
status: completed
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies: []
updated_at: "2026-11-27T16:55:56Z"
---

# T-246-record-approved-post-trial-contracts-and-task Record approved post-trial contracts and task package

## Description

Materialize the maintainer-approved post-trial fix package (reviewed revision 4;
revision 5 records the unchanged final review receipt) before product work.
The maintainer explicitly requested committing the tasks and spec changes first.
This documentation-only checkpoint discharges F1's spec-commit prerequisite;
it does not complete F1's implementation or authorize hosted calls.

## Acceptance

- Commit the exact approved per-card, nested-output and library-neutral client
  clarifications into the owning v0.5.0 headings, with no runtime-recovery exception.
  Record the resulting spec SHA256; F1 records this commit before implementation.
- Create complete F1/F2/M1/A1 task bodies through supported Taskrail import,
  preserving the reviewed fixed sequence and receipt/approval gates. Keep R1/M2
  conditional task bodies in the committed plan until their prerequisites hold.
- Install the supported T201 blocking latch naming real A1 and M2 if adopted,
  validate its actual blocked status, and preserve T200/all existing dependencies.
  Add the reviewed dated T201 acceptance-body amendment separately and hash it.
- Run git diff --check, Taskrail validate and applicable documentation/spec checks.
  Verify task bodies/dependencies against the import draft and review receipt,
  then run Taskrail verify/complete for this planning task only.
- Commit and push this planning/spec state on main, subscribe to exact-head CI,
  and confirm CI before starting product implementation. Preserve unrelated local
  changes. Record actual delivery/evidence, not planned checks as passes.

## Verification Notes

Documentation/planning-only checks apply under repository guidance. No application,
tests, dependencies, executable configuration, private inputs or runtime policies
change in this task. See docs/v0.5.0-post-trial-plan.md for the committed traceability,
approval and conditional task bodies. A completed planning task is not product QA.

## Implementation Notes

- Planning-only validation: git diff --check passed, Taskrail validate reported
  state valid, and uv run pytest -q tests/unit/install_docs_test.py passed all
  three tests. No executable inputs changed; the full Python gate is not required
  for this documentation-only checkpoint under repository guidance.
- Dedicated simplification made no edits. Independent medium-risk general/prose
  review and one fresh batched verifier found two wording defects; both were
  corrected and independently verified. No material findings remain. The review
  coordinator reconciled the specialist reports; its earlier self-passes do not
  count as independent coverage. The coordinator checked approved artifact
  revision 5 directly; source reviewers did not retrieve the approval artifact.
- Planning/spec commit 3147ad29bf11d776cd2b3243d67a14bb90359acf is on main.
  Exact-head Build run 46037434017 and CodeQL run 46037433522 passed. Runtime
  suites were correctly skipped for this documentation-only change; no runtime
  pass is claimed. Product implementation had not started at this checkpoint.
- 2026-11-27T16:53:29Z: verification pass
- 2026-11-27T16:55:56Z: Planning verification passed at 2026-11-27T16:53:29Z. Commit 3147ad29bf11d776cd2b3243d67a14bb90359acf is on main; exact-head Build and CodeQL passed. Product implementation remains separate.
