---
id: T-086-render-translation-and-ambiguity-aware-phrase
title: Render translation and ambiguity-aware phrase recognition recipes
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#translation-and-form-recognition-recipes
dependencies:
    - T-085-generate-bounded-meaningful-deck-phrases-from
    - T-065-configure-grammatical-terminology-language
updated_at: "2026-10-09T20:05:15Z"
---

# T-086-render-translation-and-ambiguity-aware-phrase Render translation and ambiguity-aware phrase recognition recipes

## Description

Provide separately selectable translation and designated-form recognition recipes
over phrase candidates. Use shared canonical claims and terminology mapping.
Export integration belongs to T-088.

## Acceptance

- Translation prompts stay Latin-first, with reviewed German reference answers and
  self-assessment, never exact-string grading or German-to-Latin production. CLI
  control language remains English.
- Recognition answers expose applicable case, number, gender, person, tense, mood
  and voice, their contribution to the construction, concise German explanations
  and supporting translations.
- Prompts identify the designated token/constituent and primary tested distinction.
  Persist that target/distinction in the semantic exercise key: questions about puella
  and amicum in the same clause are distinct exercises, not wording variants.
- Preserve legitimate alternate readings/translations. Contrast isolated amico with
  cum amico; retain ambiguity in liber amici where context permits alternatives.
  Withhold an exercise whose intended distinction cannot be taught honestly.
- Fixture expectations distinguish isolated morphological possibilities from viable
  contextual readings; amici being nominative plural in isolation does not establish
  that reading in liber amici. Include genuinely unresolved dona bona with reviewed
  alternatives; withhold a single-case question rather than inventing disambiguation.
- Preview complete prompts/answers, explanations, alternatives and withholding reasons;
  unreviewed content is clearly proposed rather than asserted.
- Persist recipe selections/limits; do not generate both recipes or all variants
  automatically. Reruns are reproducible.
- Pass the mandatory ruff/mypy/pytest chain.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- Own the phrase recipe/schema/slot/reference-template contract before rendering.
  Append any new slots without renumbering or repurposing existing ordinals, fields
  or templates; version the layout and test old-slot compatibility. T-087 persists
  reviewed objects/approvals; T-088 owns destination setup/binding and application.
