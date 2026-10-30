---
id: T-179-specify-low-review-forms-and-contextual-grammar
title: Specify low-review forms and contextual grammar releases
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#grammar-exercise-library
dependencies: []
updated_at: "2026-10-30T21:51:04Z"
---

# T-179-specify-low-review-forms-and-contextual-grammar Specify low-review forms and contextual grammar releases

## Description

Author the inactive planning split in specs/v0.5.0.md#grammar-exercise-library
and specs/v0.5.5.md, with the release order in specs/README.md. This task
records the interview decisions and proposed technical design checkpoints;
it does not implement features, activate a spec or authorize release.

## Acceptance

- v0.5.0 owns translation-first forms and controlled word groups, optional
  recognition/Latin clozes, library and blind LLM verification, automatic
  acceptance, one CLI workflow, reports, Anki status and morphology themes.
- Keep curated grammar/mnemonics in v0.5.0 with separate human approval of
  both new rule-mnemonic pairs and rendered cards.
- v0.5.5 reuses the shared pipeline for original-corpus sentence functions
  and constructions, supporting deck-driven and reading-driven entry points.
- Preserve local/hosted explicit configuration, no silent fallback, optional
  unresolved-item review, material-yield explanations and bounded selection.
- Mark schema, policy, evaluation and release-contract decisions as proposed
  design prerequisites; include acceptance/checkpoint ownership without
  claiming materialized implementation tasks or readiness.
- Independently review the spec and complete local planning package; record
  actual structural validation and unresolved findings.
- Keep v0.4.0 and its activation unchanged by this task. Do not modify
  application code or execute future implementation tasks.

## Verification Notes

- Inspect both specs, the index, anchors and active-spec state. Run
  `mise exec -- taskrail validate` and `git diff --check`.
- Independent planning reviews covered the local candidate at base revision
  0e82dc7afbdcb4c461506d941b0431a1aed55ac0 plus the spec/index/authoring-task
  changes. Budget: two initial gates and at most two targeted rechecks total.
- This is documentation-only work. Runtime/library/model and native Anki
  evidence described in the specs remains future release work.

## Implementation Notes

- Local specification authoring only; task lifecycle remains todo. No spec
  activation or implementation-task creation occurred. Taskrail regenerated
  STATE.md to count this task; the active spec remains v0.4.0.
- Gate 1 initially found missing installed-card invalidation handoff and
  processing-budget deferral reporting. Both were corrected; targeted
  recheck 1 passed. Curated presentation approval was distinguished from
  linguistic approval. Gate 2 passed the bounded planning package and its
  proposed dependency/evidence mapping, not implementation readiness.
- Review budget consumed: two initial gate passes and one targeted recheck.
  Reviewed spec content SHA-256 values:
  - specs/v0.5.0.md: 76e2d351fdfaceb838ed947101eac41ee356cb710577a342e047b5e3b86afc6b
  - specs/v0.5.5.md: 62b7da0a02523dce7f098ca0c4f7244a426bdb2379889eee0b55eb36c3b587ed
- Before delivery, main was fast-forwarded to
  535100f8988a8ef1fe3cd4d3092b21285b74fbb8. The two reviewed specs are
  byte-identical; the index preserves upstream v0.4.0 documentation changes.
  Taskrail allocated T-179 instead of the original local T-175, which had
  been used on origin/main. No existing task or upstream state was replaced.
- The maintainer subsequently authorized committing and pushing these
  documentation changes on main; no feature implementation was authorized.
- Follow-up interview decisions revise both inactive specs: process all
  selected material within processing limits and export all eligible cards
  after duplicate/redundant-variant suppression by default, with smaller
  batches optional and valid stored checks reused. No default card quota.
- Reliability takes priority over yield. Keep uncertain candidates outside
  Anki without mandatory user review; show acceptance status, collapsible
  evidence/limitations and filterable metadata, not uncalibrated model
  percentages on cards or in tags.
- v0.5.5 targets the broader examination-oriented clause families, subject
  to independently evaluated subtype boundaries. Use manageable sentences
  or coherent marked parts of longer sentences, preserve necessary context
  on the front and full context on the answer/report, and avoid analysis
  labels on translation fronts or automatic whole/part card multiplication.
- These follow-up edits do not activate a spec or authorize implementation,
  another commit or another push. Prior review hashes cover the delivered
  snapshot, not these semantic changes.
- The final targeted semantic-delta recheck found no blockers in the changed
  Gate 1 contracts or affected Gate 2 proposed package mapping. Total review
  budget consumed: two initial gates and two targeted rechecks; none remain.
  This establishes draft consistency, not implementation readiness or
  measured linguistic reliability. Follow-up reviewed SHA-256 values:
  - specs/v0.5.0.md: 615fab4f8e4f2001ef9d5e206dd56ab402c6379256629c7c97671a011320189c
  - specs/v0.5.5.md: 159259a7089de6661721b69b154181f8dbe23dccf8d90a7383bde66238e8ab4d
- Follow-up structural checks passed: `mise exec -- taskrail validate` and
  `git diff --check`. No application code or templates changed; runtime and
  native Anki acceptance scenarios remain planned evidence, not executed tests.
