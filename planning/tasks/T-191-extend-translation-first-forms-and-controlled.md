---
id: T-191-extend-translation-first-forms-and-controlled
title: Extend translation-first forms and controlled phrase recipes
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#grammar-exercise-library
dependencies:
    - T-190-gate-complete-card-assertions-on-accepted-evidence
updated_at: "2026-11-02T23:10:11Z"
---

# T-191-extend-translation-first-forms-and-controlled Extend translation-first forms and controlled phrase recipes

## Description

P50-3 extends phrase_inputs/phrase_recipes and existing form helpers, not a second
phrase engine. Own bounded patterns and one-objective recipe rendering. Follow
approved contracts and spec execution-isolation/final-validation requirements.

## Acceptance

- Cover supported verb forms/production/recognition and stems, noun nominative/genitive
  pairs and declension/case/number, adjective agreement, preposition-noun, noun-genitive,
  adjective-noun and verb-noun mini-clauses. Government, agreement, sense and valency
  constrain generation; unsupported combinations are reported.
- Automatically generate and export isolated perfect passive participle (PPP)
  translation cards from confirmed source-deck PPP evidence when shared acceptance
  checks pass, without per-card human approval. PPP recognition is selectable;
  Latin-form production/form-cloze variants remain opt-in. This is required v0.5.0
  scope, not deferred to v0.5.5.
- Keep PPP distinct from supine. An unconfirmed fourth principal part or an ending
  alone cannot establish PPP status. Missing, contradictory or ambiguous evidence
  follows the shared withholding/review policy; preserve legitimate readings or
  withhold unanswerable recognition prompts.
- Translation is primary, recognition selectable, Latin production/form-cloze opt-in.
  Default output has no Latin-form clozes or multiplication across every recipe.
- Contrasting cases, verbs and agreement produce distinct answerable objectives.
  Preserve alternate isolated readings or withhold unanswerable distinctions.
  Translation fronts do not leak analysis; references are guidance, not grading strings.
- Retain pattern-supplied basic roles without corpus syntax discovery/v0.5.5 work.
  Preserve identities/recipes outside explicit opt-in upgrades.

## Verification Notes

Exercise public preview/export with wrong government/agreement/valency, ambiguous
prompts and default/opted-in recipes in German/English. Check semantic keys and
existing characterization goldens. Run the mandatory validation chain.

For PPP, contrast confirmed PPP and supine profiles, an unconfirmed fourth principal
part, and missing/contradictory evidence. Verify eligible PPP translation export and
selectable recognition, default exclusion and explicit opt-in of production/clozes,
and withheld unsupported claims. No corpus construction discovery is required.

## Implementation Notes
