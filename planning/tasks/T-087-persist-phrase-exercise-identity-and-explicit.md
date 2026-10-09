---
id: T-087-persist-phrase-exercise-identity-and-explicit
title: Persist phrase exercise identity and explicit linguistic approvals
status: todo
priority: high
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies:
    - T-086-render-translation-and-ambiguity-aware-phrase
updated_at: "2026-10-09T20:05:15Z"
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
