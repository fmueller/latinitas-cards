---
id: T-160-build-the-accepted-fact-view-and-consume-latinitas
title: Build the accepted-fact view and consume latinitas-morph/1
status: todo
priority: high
spec_ref: specs/v0.4.0.md#context-and-translation-recipes
dependencies:
    - T-156-add-the-language-table-and-english-terminology
updated_at: "2026-10-27T23:10:48Z"
---

# T-160-build-the-accepted-fact-view-and-consume-latinitas Build the accepted-fact view and consume latinitas-morph/1

## Description

Implements Fixed Contracts **`latinitas-morph/1`** and **Fact rules** as one reusable
read-only view over a corpus, its sidecar and claims file (building on
`corpus_claims.current_annotations` and `join_corpus_decisions`): per passage, current
segmentation, sentences with fragment flags, and per anchor the facts with review state
and fact-state reason (`missing_fact`, `stale_fact`, `open_fact`, `withheld_fact`,
`conflicting_facts`, `unsupported_morphology`, `wrong_language`, `no_features`). v0.3.0
annotation validation stays unchanged; morph values are validated only here.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Accepted, open, withheld, stale, orphaned, superseded and retracted annotations
  resolve to the documented states; spanned facts require the current
  `segmentation_id` and exact token or sentence spans; span-less translations need only
  a current `passage_hash`.
- `latinitas-morph/1` validation: lemma, pos, finite-only `mood`, `degree`, `verbform`;
  another tagset or invalid value yields `unsupported_morphology`; an anchor not equal to
  one word token is unsupported.
- `conflicting_facts` applies after the language filter, covering same-anchor values and
  a one-sentence passage's sentence-span versus span-less translation; identical values
  count once; a German and an English translation on one span do not conflict for an
  `en` profile.
- No test imports CLTK, Stanza or an LLM client.

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
