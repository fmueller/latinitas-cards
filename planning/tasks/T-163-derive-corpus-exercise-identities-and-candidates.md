---
id: T-163-derive-corpus-exercise-identities-and-candidates
title: Derive corpus exercise identities and candidates for the four recipes
status: completed
priority: high
spec_ref: specs/v0.4.0.md#corpus-first-profiles-and-exercise-identity
dependencies:
    - T-160-build-the-accepted-fact-view-and-consume-latinitas
    - T-162-define-the-corpus-generation-profile-and-corpus
updated_at: "2026-10-29T09:39:58Z"
---

# T-163-derive-corpus-exercise-identities-and-candidates Derive corpus exercise identities and candidates for the four recipes

## Description

Implements Fixed Contracts **Recipes** and **Exercise identity**: candidate
generation per occurrence, eligibility from the accepted-fact view, units (sentence,
passage, fragment rules), targets keyed by folded surface and occurrence over all word
tokens (phrase windows over word tokens only), `object_key`, `LatinitasID`, card
semantic keys and `context_digest`. Rendering, ranking and selection are later tasks.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Independently specified identity vectors in `tests/unit/identity_vectors_test.py`
  lock one exercise per recipe in `de` and `en`; existing vectors are unchanged.
- Two profiles with the same corpus and language yield identical IDs; `de` and `en`
  differ; gloss, translation, lemma, theme and known-lemma changes keep IDs.
- With two tokens of one form, accepting the second token's facts first keeps both IDs.
- A phrase `a, b` (internal punctuation) has target form `a b`; a span ending in
  punctuation (`a b,`), equal to a sentence or crossing one is ineligible
  `invalid_span`; a guided target without features is
  `no_features`; fragments are excluded for guided analysis and sentence translation.
- Passage units arise only for span-less translations on passages of 2+ sentences within
  `max_tokens`; a span-less translation on a one-sentence passage serves the sentence.
- A correction splitting sentence 0 changes the `context_digest` at index 2.

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
- 2026-10-29T09:39:53Z: verification pass
