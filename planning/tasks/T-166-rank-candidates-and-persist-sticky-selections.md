---
id: T-166-rank-candidates-and-persist-sticky-selections
title: Rank candidates with latin-rank/1 and persist sticky selections with a candidate report
status: todo
priority: high
spec_ref: specs/v0.4.0.md#ranked-card-budgets
dependencies:
    - T-163-derive-corpus-exercise-identities-and-candidates
    - T-165-read-known-lemmas-from-a-confirmed-deck-profile
updated_at: "2026-10-27T23:10:48Z"
---

# T-166-rank-candidates-and-persist-sticky-selections Rank candidates with latin-rank/1 and persist sticky selections with a candidate report

## Description

Implements Fixed Contracts **`latin-rank/1`** and **Selection and report**:
`corpus-cards select` with filters, factors (token-level `target`/`novelty` for
single-word recipes), ordering, kept-first greedy fill, budgets, near-duplicate and
per-lemma exclusions, the closed reason vocabulary, `context_changed`/`target_missing`,
`--confirm-context`, the selection and candidates schemas, writer locking/alias/atomic
rules, and `invalid_reading`/`binding_drift`/`profile_mismatch` errors.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- A hand-computed fixture pins every factor, weight, score, ordering key, outcome and
  exclusion precedence; expected values are derived by hand, not from the code.
- Recipe, work and run budgets at and just above their limits; the run-budget case
  starts with more sticky selections than `per_run`.
- Reruns keep all selections, add only up to free budget, keep newly ineligible ones
  occupying recipe/work budget; filters never remove a selection.
- In a sentence containing a target lemma, the lexical exercise highlights the target
  word although it is neither first nor lowest `latinitas_id`.
- Homograph lemmas `liber (1)`/`liber (2)` match known lemma and target `liber` but
  keep separate `per_lemma` buckets; with no known-lemma input `u` is `w` minus a
  single-word target.
- `invalid_span` entries sort after ineligible candidates by `annotation_id`.
- Asymmetric near-duplicate and per-lemma fixtures distinguish multiset from set
  arithmetic; `confidence` changes no score; no known-lemma input makes every word
  unknown and new.
- After a correction, a selected exercise is `context_changed` until `--confirm-context`.
- Corpus argument order does not change output; two subprocess runs with different
  cwd, `TZ`, locale and `PYTHONHASHSEED` give byte-identical selection and report;
  unchanged inputs are a no-op; a held lock exits 4 `busy` with no byte change; aliases
  are rejected; an injected failure before replace keeps the old selection.

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
