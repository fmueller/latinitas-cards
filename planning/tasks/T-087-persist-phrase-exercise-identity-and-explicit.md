---
id: T-087-persist-phrase-exercise-identity-and-explicit
title: Persist phrase exercise identity and explicit linguistic approvals
status: completed
priority: high
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies:
    - T-086-render-translation-and-ambiguity-aware-phrase
updated_at: "2026-10-11T15:04:12Z"
---

# T-087-persist-phrase-exercise-identity-and-explicit Persist phrase exercise identity and explicit linguistic approvals

## Description

Integrate phrase objects and both recipes with shared claim decisions, stable
note/card identities and explicit correctness/naturalness approval. T-064 owns
shared sidecars and existing recipes; this task owns phrase adapters.

## Acceptance

- Persist a separate coherent phrase object keyed by selected lexical identities/senses,
  construction and tested form choices; recipe/semantic exercise keys distinguish
  translation/recognition. Never merge source notes or attach a phrase to one word.
- Wording, gloss and theme corrections preserve object/note/card keys and slots;
  changed tested forms/meaning are distinct exercises without collisions.
- Reuse T-084/T-085 canonical constituent/candidate bindings and T-086 designated
  target/distinction keys. Test reordered input selections, reversed semantic roles,
  distinct senses sharing a gloss and different questions about the same phrase.
- Require reviewed patterns/forms and explicit per-combination correctness/naturalness
  and rendered-answer approval before eligibility. Claim attestation is not export
  or managed-plan approval.
- Automatically consume shared sidecars for phrase preview/export, including individual
  and agent-style batch acceptance, withholding and authored corrections. Changed phrase,
  analysis, translation, evidence or linguistic context invalidates relevant approvals;
  unchanged approvals survive reruns and presentation changes.
- Extend shared read-only inspection/check provenance to both phrase recipes. Show
  identities, slots, claims, fingerprints, evidence, alternatives and withholding
  reasons without Anki.
- Test identity stability, distinct form choices, stale/unmatched decisions, corrected
  answers and unchanged source notes; pass the mandatory ruff/mypy/pytest chain.
- Negative controls independently withhold eligibility for reviewed inputs without
  combination approval and combination approval without rendered-answer approval.
  A changed answer cannot reuse its old answer approval.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- Reused canonical role/source/sense/form/pattern bindings and target/distinction
  keys. Each question derives a separate coherent note identity from the composite
  phrase source; shared phrase object provenance and slots 10/11 remain explicit.
- Recipe-plan T064 sidecars independently attest combination correctness/naturalness
  and typed rendered references, including individual, batch and authored correction
  decisions. Preview and shared read-only inspection/check consume them automatically.
  Linguistic context changes invalidate relevant approvals; presentation and reruns
  preserve them. Source notes remain unchanged; attestation is not payload approval.
- Strict RED/GREEN covered absent CLI adapters, independent withholding state,
  explicit alternative provenance and duplicate withholding banners. A deliberately
  regressed combination-evidence binding failed the strengthened multi-question test;
  restoring scoped constituent/pattern evidence passed it.
- Dedicated code-simplifier indexed claim lookup. Independent General, Python,
  Security and Pydantic/FastAPI code-reviewer lanes ran, followed by fresh candidate
  validation and disposition verification. The sole candidate was rejected because
  combination evidence was already scoped independently of question references;
  strengthened tests and real CLI acceptance corroborated the rejection. No validated
  findings or unresolved task-local issues remained.
- Real subprocess CLI acceptance exercised individual/batch/corrected/withheld
  decisions, both independent missing gates, stale translation/analysis/evidence,
  inspection/check, reruns and conflict atomicity. Six generated-field Chromium
  render states were captured at 2x and inspected; duplicated withholding was fixed.
- CSV/destination application remains T088. Native automatic appearance acceptance
  on Desktop/Mobile is pending, not passed or covered; maintainer deferral dated
  2026-10-11 removes only its implementation-loop blocker. Existing native checklist
  and limitations remain intact; browser rendering is not a substitute. T089 is
  outside this task's ownership. No additional task was implemented.
- 2026-10-11T15:03:59Z: verification pass
- 2026-10-11T15:04:12Z: Implementation verified pass at 2026-10-11T15:03:59Z after final ruff/mypy/pytest (1016 passed), simplifier, independent lanes, fresh candidate/disposition verification, real CLI acceptance and inspected Chromium renders. Native Desktop/Mobile automatic appearance acceptance remains pending; maintainer 2026-10-11 deferral applies only to implementation-loop blocking. T088 application and T089 remain outside this task.
