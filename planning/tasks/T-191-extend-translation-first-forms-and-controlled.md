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

## Implementation Notes
