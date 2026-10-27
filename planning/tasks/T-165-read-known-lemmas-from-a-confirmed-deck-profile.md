---
id: T-165-read-known-lemmas-from-a-confirmed-deck-profile
title: Read known lemmas from a confirmed deck profile
status: todo
priority: medium
spec_ref: specs/v0.4.0.md#ranked-card-budgets
dependencies:
    - T-162-define-the-corpus-generation-profile-and-corpus
updated_at: "2026-10-27T23:10:48Z"
---

# T-165-read-known-lemmas-from-a-confirmed-deck-profile Read known lemmas from a confirmed deck profile

## Description

Implements Fixed Contracts **Known-lemma input**: load any supported confirmed deck
profile and its source read-only, extract and fold one lemma per record (HTML stripped,
punctuation trimmed), report unreadable records, and expose lemma-to-deck-source-identity
links and digests for ranking and the report. Never writes deck files or state.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Fixtures with `dīcō, dīcere`, HTML markup and macrons yield the folded lemma
  `dico`; unreadable records are reported and skipped.
- CSV (with scope state) and APKG fixtures produce identical lemma sets for identical
  content; source and profile digests are reported; source files are byte-unchanged.

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
