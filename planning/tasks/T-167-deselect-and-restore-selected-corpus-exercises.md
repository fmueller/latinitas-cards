---
id: T-167-deselect-and-restore-selected-corpus-exercises
title: Deselect and restore selected corpus exercises
status: todo
priority: medium
spec_ref: specs/v0.4.0.md#ranked-card-budgets
dependencies:
    - T-166-rank-candidates-and-persist-sticky-selections
updated_at: "2026-10-27T23:10:48Z"
---

# T-167-deselect-and-restore-selected-corpus-exercises Deselect and restore selected corpus exercises

## Description

Implements `corpus-cards deselect` with `--reason` and `--restore` per Fixed
Contracts **Selection and report** and **Commands**: deselected entries keep the full
selected entry plus reason and run; restore returns it with its stored context digest;
`unknown_exercise` for IDs in neither list; no-op cases; writer rules.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Deselect then `select` never re-selects; `--restore` makes it kept again with the
  stored `context_digest` and `run` set to the restoring run (a correction made while deselected yields `context_changed`).
- Unknown ID exits 3 `unknown_exercise`; repeated deselect or restore of a selected ID is
  a byte-identical no-op; `runs` increments only on change.

## Implementation Notes

- Start only when `taskrail status --json` shows active spec `specs/v0.4.0.md`; otherwise
  stop without `taskrail start` and report that v0.4.0 is not active.
- Follow-up rule (as in v0.3.0): (a) a defect in completed work this task builds on is
  fixed inside this task with a regression test; (b) follow-ups are filed only for work
  no remaining v0.4.0 task needs, with `--priority medium` or `high`, never `low`, and
  never with `--follow-up` on T-172; (c) a follow-up needing a maintainer decision (for
  example a Fixed Contracts disagreement) is created and then blocked with the reason
  "maintainer decision: ...".
- Register every new diagnostic code in `cli_contract.py` and add its row to the
  `docs/agent-cli-contract.md` table in this task (`cli_contract_test.py` compares them).

## Verification Notes

- Not started.
