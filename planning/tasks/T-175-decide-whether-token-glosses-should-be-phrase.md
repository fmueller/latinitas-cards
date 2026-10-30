---
id: T-175-decide-whether-token-glosses-should-be-phrase
title: Decide whether token glosses should be phrase translation candidates
status: completed
priority: medium
spec_ref: specs/v0.4.0.md#context-and-translation-recipes
dependencies:
    - T-163-derive-corpus-exercise-identities-and-candidates
updated_at: "2026-10-30T20:23:07Z"
---

# T-175-decide-whether-token-glosses-should-be-phrase Decide whether token glosses should be phrase translation candidates

## Description

Follow-up found during T-163 review. Fixed Contracts **Recipes** generates "one
candidate per current spanned translation annotation whose span is not exactly a
sentence span" for `corpus_phrase_translation`, and a span that is not 2-8 word tokens
is ineligible `invalid_span`. Lexical recognition needs "a translation anchored to
exactly that token", so every reviewed token gloss (in any language) is also reported
as an `invalid_span` phrase entry whenever a profile enables phrase translation. T-163
implements the contract literally (`corpus_exercises.py`, pinned by
`corpus_exercises_test.py`). The candidate report (T-166) would then carry one
`invalid_span` row per gloss, which looks like noise rather than a reviewer-actionable
span problem. Changing which annotations are phrase candidates changes the binding
Recipes contract, so it needs a maintainer decision: either one-word-token translation
spans are excluded from phrase candidates, or the per-gloss `invalid_span` entries are
documented as intended.

## Acceptance

- The maintainer records which behavior is intended for one-word-token translations.
- If they are excluded, a token gloss yields no phrase candidate while a non-qualifying
  multi-token span still yields `invalid_span` per `annotation_id`, and identities of
  every other candidate stay unchanged.
- Otherwise the docs state that token glosses appear as `invalid_span` report entries.

## Verification Notes

- Verification passed at 2026-10-30T20:22:47Z after the exact Ruff, mypy and
  pytest chain: 3180 passed, 7 skipped. Baseline regression checks failed as
  intended; the final candidate's focused suite passed all 81 tests.
- Dedicated simplification and independent medium-risk conventional review,
  including specialist and adversarial passes, found no remaining defects. The
  full-suite renderer fixture assumption was corrected and independently
  rechecked without weakening its identity-error assertion.
- Independent integrated acceptance reran normalize, annotation review and
  select/report/preview workflows against the final candidate and the base.
  In both languages, only six exact-word gloss report rows disappeared; all
  other rows, exercise identities and previews stayed unchanged, including
  lexical answers. Malformed spans retained individual annotation IDs.
- Four inside-token malformed controls were persisted directly after both
  versions' patch planners rejected them; downstream review and consumers were
  exercised normally. Other fixture records passed plan/approve/apply.

## Implementation Notes

- 2026-10-29T09:28:32Z: maintainer decision: whether translation annotations anchored to one word token (lexical glosses) should be excluded from phrase translation candidates, which the Fixed Contracts Recipes candidate rule currently reports as invalid_span
- 2026-10-30T19:58:08Z: Maintainer approved: exclude translations anchored to exactly one word token from phrase candidates because they belong to lexical cards; retain invalid_span per annotation_id for malformed multiword spans and preserve all other candidate identities. Authorize only the necessary narrow Fixed Contracts clarification.
- 2026-10-30T20:22:47Z: verification pass
- 2026-10-30T20:23:07Z: Implement the approved exact-word gloss exclusion with lexical preservation and unchanged remaining identities; independent review and final-candidate integrated acceptance passed after full mandatory gates. No successor started.
