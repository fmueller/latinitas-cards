---
id: T-169-plan-managed-updates-for-corpus-exercises
title: Plan managed updates and blocked retirement for corpus exercises
status: todo
priority: high
spec_ref: specs/v0.4.0.md#corpus-first-profiles-and-exercise-identity
dependencies:
    - T-168-preview-export-and-inspect-corpus-exercises
    - T-167-deselect-and-restore-selected-corpus-exercises
updated_at: "2026-10-27T23:10:48Z"
---

# T-169-plan-managed-updates-for-corpus-exercises Plan managed updates and blocked retirement for corpus exercises

## Description

Implements `corpus-cards plan` per Fixed Contracts **Commands**: a `managed plan`
request like `phrases plan`, with approved proposals for eligible selections, withheld
proposals carrying anchored fields, `Profile` copied from the anchored note, retire
proposals only for explicitly deselected IDs, and unsupported create for missing notes.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- On a synthetic adopted snapshot: a reviewed translation correction yields an
  approvable retained-slot update that `managed approve` and `emit` accept; a passage
  correction re-reviewed and confirmed yields a `Passage` content update.
- `plan` exits 4 `binding_drift` and 3 `profile_mismatch` like the other commands;
  ineligible anchored selections become `withheld` proposals with anchored fields;
  ineligible or eligible exercises outside the bound managed set are reported as not
  imported, and only bound-set members missing from the destination become unsupported
  `create` operations.
- Ineligible selections change nothing; a deselected ID yields a blocked retire effect;
  no undeselected note is ever proposed for retirement; a note shared with another
  profile gets no `Profile` operation.
- A target-language switch with a new selection file proposes no retirement of
  old-language notes until they are deselected in the old selection.

## Implementation Notes

- Start only when `taskrail status --json` shows active spec `specs/v0.4.0.md`; otherwise
  stop without `taskrail start` and report that v0.4.0 is not active.
- Follow-up rule (as in v0.3.0): (a) a defect in completed work this task builds on is
  fixed inside this task with a regression test; (b) follow-ups are filed only for work
  no remaining v0.4.0 task needs, with `--priority medium` or `high`, never `low`, and
  never with `--follow-up` on T-172; (c) a follow-up needing a maintainer decision (for
  example a Fixed Contracts disagreement) is created and then blocked with the reason
  "maintainer decision: ...".

## Verification Notes

- Not started.
