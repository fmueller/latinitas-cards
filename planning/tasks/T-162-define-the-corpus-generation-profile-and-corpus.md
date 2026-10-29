---
id: T-162-define-the-corpus-generation-profile-and-corpus
title: Define the corpus-generation profile and the corpus-cards command group
status: completed
priority: high
spec_ref: specs/v0.4.0.md#corpus-first-profiles-and-exercise-identity
dependencies:
    - T-156-add-the-language-table-and-english-terminology
updated_at: "2026-10-29T08:46:13Z"
---

# T-162-define-the-corpus-generation-profile-and-corpus Define the corpus-generation profile and the corpus-cards command group

## Description

Implements Fixed Contracts **Corpus profile**: the strict `latinitas-corpus-profile`
v1 model with defaults and bounds, `corpus-cards profile validate`, the new contract
group `corpus-cards` (added to the matrix walk tuple), the committed JSON Schema, and the
shared `profile_mismatch` check used by later commands. Reading ranges are validated
against corpora only at use (`invalid_reading`, later tasks).

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- The documented example validates; unknown fields, floats, a non-`de`/`en` language, an
  unknown or duplicate recipe, out-of-bound budgets, limits violating `1 <= min <=
  ideal <= max <= 200`, duplicate or unsorted corpus IDs and a reading `corpus_id` not in
  `corpora` exit 3.
- `profile validate --json` emits one schema-valid envelope; matrix rows and the
  diagnostic registry cover the group; the schema drift test passes.
- The profile digest is the canonical structured digest and is stable across runs.

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
- 2026-10-29T08:46:07Z: verification pass
