---
id: T-179-specify-low-review-forms-and-contextual-grammar
title: Specify low-review forms and contextual grammar releases
status: completed
priority: medium
spec_ref: specs/v0.5.0.md#grammar-exercise-library
dependencies: []
updated_at: "2026-11-18T12:23:37Z"
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

## Historical delivery reconciliation (2026-11-18)

The maintainer authorized reconciling T-179's historical delivery and continuing
the loop while keeping judge runs paused. This cycle records the delivered
planning work only. The original description, acceptance criteria, verification
notes and implementation history above remain historical statements, including
the deliberately retained todo lifecycle and then-active v0.4.0. They are not
descriptions of today's approved sources or runtime implementation.

### Delivered identities and acceptance mapping

Starting base and fetched origin/main:
`f97d9c005d9162559e2e0193cdafc5837d4c6f17`.
Both original deliveries are ancestors of that base:
`6f2aab703b08003cf6b685e06526015f433155e1` (initial planning split) and
`893cef4d31692be71798d6832ece999dd885522e` (follow-up decisions).
Recomputing both spec files at each commit reproduces all four SHA-256 values
recorded above. These hashes identify those historical snapshots, not current
spec bytes. The initial commit changes only the two specs, index, T-179 and
generated STATE; the follow-up changes only the two specs and T-179.

Acceptance mapping below refers to the final historical snapshot at
`893cef4d31692be71798d6832ece999dd885522e`, not to implemented features:

1. `specs/v0.5.0.md`: Grammar Exercise Library, Shared Evidence And Automatic
   Acceptance, CLI Orchestration And Provider Boundaries, Outcome Reports And
   Anki Acceptance Metadata, and Semantic Color Layers specify translation-first
   forms/controlled phrases, opt-in recognition/clozes, verification, orchestration,
   reports, status and morphology presentation.
2. Its Curated Grammar And Mnemonics section requires separate human approval
   of rule-mnemonic pairs and rendered cards; automatic acceptance cannot bypass it.
3. `specs/v0.5.5.md`: Summary, Planning Status And Dependencies, Deck-Driven Corpus
   Examples and Reading-Driven Exercises specify original-corpus constructions
   through both entry points and reuse of v0.5.0's pipeline. The historical index
   places inactive v0.5.5 after inactive v0.5.0 and before v0.6.0.
4. The historical provider, evidence, reporting and selection sections preserve
   explicit local/hosted configuration, no silent fallback, optional unresolved
   review, yield explanations, processing bounds and full eligible export by
   default without a card quota; optional batching reuses valid checks.
5. Both specs' Proposed Delivery And Acceptance Checkpoints and Open Technical
   Design Checkpoints name prerequisite schemas/policies/evaluation, future task
   evidence ownership and maintainer decisions, not ready implementation tasks.
6. The original notes record two initial planning gates, the initial handoff/
   deferral findings and fixes, and two targeted rechecks, ending without blockers
   for that draft. They also record passed Taskrail/diff structural checks.
   This is the preserved historical review evidence, not a new planning review
   or empirical acceptance. Its review budget remains exhausted; none is reset.
7. Neither delivery commit touches v0.4.0, application code or tests. The initial
   delivered STATE keeps v0.4.0 active; both specs and the index remain inactive
   planning at that time. No later implementation is attributed to T-179.

### Subsequent supersession and limits

Activation and implementation permission came later, separately: T-184 activation
at `50c8b7752f17186534d27164cab3f811063fd9df`; exact design adoption at
`568f02cf491e6252e795e10474833bd6a137fc21`; T-185 design verification at
`b5d56a04946225d3345c9b85d3b88a31ebc3c404`; and T-186's separate maintainer
engineering-approval record at `c5d7ec92adb8379d51cb4f541d39dc5576003255`.
That approval adopts exact sources for engineering, not empirical/numeric policy
or automatic enablement. T-187 through T-190 subsequently record their own
runtime implementation, checks and limits; they are already delivered at this
cycle's base. Those later records supersede historical planning/permission
statements only in their stated scope. Earlier source identities, failures,
exhausted allocations and approval boundaries remain unchanged.

This reconciliation adds no spec authoring, activation, design gate, provider or
judge/study run, private evidence access, linguistic reliability claim, release
readiness, publication or automatic-policy permission. T-205 and T-207 remain
unchanged. Only T-179 recording/lifecycle and Taskrail-generated STATE may change;
the outer loop owns independent acceptance and successor selection.

### Recording-cycle verification and review

- Structural checks passed on the recording candidate: Taskrail validation,
  `git diff --check`, original-body and unchanged metadata preservation,
  two-file scope, active-task ownership, pinned spec/anchor and local links,
  six cited commit ancestries, four historical hashes, original delivery file
  lists and the separately completed T-187 through T-190 records.
- Dedicated simplifier readback made no edits. Independent low-risk conventional
  review used a separate general reviewer and coordinator reconciliation/self-
  challenge; both found no recording defects, with no dispositions to fix or
  defer. The approved local diff SHA-256 was
  `2a8acad1ee865744c12468e433a2bbb74198a1e5be8487f99f3b96dc2e1e2c45`.
  This paragraph and supported lifecycle bookkeeping follow that reviewed diff;
  they do not change its historical claims or reopen planning reviews.
- Human-message provenance and earlier reviewer execution were not independently
  authenticated; this cycle uses the supplied maintainer authorization and
  committed historical records. STATE validation checks consistency, not
  generator provenance. No private evidence or empirical results were examined.
- Only narrative/lifecycle inputs changed. Searches found no documentation
  contract tests consuming these changed notes, so runtime TDD, the full Python
  chain and manual runtime/Anki checks do not apply and were not run. Final
  structural checks and recording readback are required after lifecycle writes.
- 2026-11-18T12:23:28Z: verification pass
- 2026-11-18T12:23:37Z: Historical planning delivery reconciled; recording-only verification passed 2026-11-18T12:23:28Z after dedicated simplifier and independent low-risk review. Original text, identities, failures and exhausted planning budget preserved. Later activation/engineering approval and runtime implementation remain separate; judges paused, no policy or publication authority. Outer loop owns independent acceptance and successor selection.
