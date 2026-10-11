---
id: T-091-apply-configured-appearance-to-both-phrase-recipes
title: Apply configured appearance to both phrase recipes
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#automatic-lightdark-appearance
dependencies: []
updated_at: "2026-10-11T18:41:19Z"
---

# T-091-apply-configured-appearance-to-both-phrase-recipes Apply configured appearance to both phrase recipes

## Description

ROUND1 confirmed that both phrase recipes ignore configured appearance and
palette in their generated presentation at revision
6eb490102cb167404b4534015e9d5f3d79447763. Six reviewed exports (muted/monochrome
times light/dark/auto) have identical phrase prompt/answer HTML, using bare
`<div>` elements. The declared CSS v3 theme selectors require `.morphology-v1`,
which is absent. Twelve exact-template Chromium renders stay near-black on white,
including fixed dark and auto under `body.card.nightMode`.

Impact: medium. Selecting dark or automatic appearance does not recolour phrase
cards; the supported palette setting is not applied. This is distinct from
inspection text fidelity (T-090). The pinned
`specs/v0.2.1.md#automatic-lightdark-appearance` requires generated markup and
reference CSS to honor these settings and documents no phrase exclusion.
Parent scope adjudication on 2026-10-11 confirms this integration defect is in
scope; native verification deferral does not defer browser-verifiable markup.

## Acceptance

- Both phrase translation and designated-form recognition outputs apply the
  effective fixed light/dark or auto presentation and both supported palettes
  through the shared declared reference CSS. Fixed settings remain fixed under
  opposing client classes; auto has readable light fallback and follows each
  supported night-mode class through day/night/day without regeneration/reimport.
- Public preview/export and rendered exported-card tests exercise both recipes,
  both palettes and fixed/auto states. Assert actual content-panel computed
  presentation, not merely stored profile values or an invented harness wrapper.
- Presentation-only changes preserve canonical claim fingerprints, accepted
  reviews, coherent note/card identities, semantic keys, eligibility and slots.
  Changed generated payloads still require a fresh managed plan and exact payload
  approval; stale receipts must not authorize retained-slot updates.
- Keep Latin-first fronts free of answer translations/analyses; preserve reviewed
  German answers, escaped literal prose, safe markup and no JavaScript.
- Integrate with the declared shared reference CSS and existing phrase layout.
  Do not silently rebind templates, change scheduled ordinals or bypass old
  digest rejection. Any necessary layout/style change requires the existing
  explicit manual setup/reconciliation boundary, not an implicit migration.
- Do not add exhaustive morphology, segmentation, arbitrary template emulation,
  new linguistic claims or native-verification claims to this repair.
- Preserve the exact pending Desktop/AnkiMobile scope in
  `docs/morphology-native-verification.md`; browser evidence is supporting only.
- Run the mandatory ruff, mypy and pytest chain.

## Verification Notes

- ROUND1 evidence dated 2026-10-11 uses ordinary lexical/source/profile files,
  individual reviewed claims, four bounded phrases and eight explicit exercises.
  The decisive clause is `puella amicum videt`, with translation
  `das Mädchen sieht einen Freund` and recognition target `videt [verb]`.
  No internal application models were used.
- Reproduce from any freshly reviewed fixture following
  `docs/reviewed-phrase-inputs.md`, enabling both recipes. In a disposable copy,
  keep input/review files unchanged and set its referenced profile's
  `morphology.theme` to each of `muted`/`monochrome` and
  `morphology.appearance` to each of `light`/`dark`/`auto`. For each combination:

  ```sh
  uv run latinitas-cards phrases export inputs.json selected.json recipes.json NEW.csv --json
  uv run latinitas-cards claims inspect inputs.json --selections selected.json \
    --recipes recipes.json --artifact NEW.csv --check --json
  ```

  Inspect the four `PhraseTranslationPrompt`, `PhraseTranslationAnswer`,
  `PhraseRecognitionPrompt`, `PhraseRecognitionAnswer` fields using a CSV parser.
  Actual: all six combinations have equal payloads, no morphology/theme wrapper,
  and all source-bound checks return zero. Example prompt:
  `<div>puella amicum videt<br>Bestimme videt [verb]: Analyse.</div>`.
  Expected: generated presentation applies the selected reference-CSS settings
  without altering linguistic claims or exercise identity.
- Render the exported fields with the exact declared phrase-v1 templates
  (slots 10/11) and complete CSS v3, under `body.card` and
  `body.card.nightMode`, without adding styling wrappers. Actual in all twelve
  captures: `.morphology-v1` count 0, inherited foreground `rgb(26, 26, 26)`,
  background `rgb(255, 255, 255)`, zero scripts, readable but unchanged prose.
  Expected: the selected content palette, including dark auto/night presentation.
- Original synthetic audit is retained in the ROUND1 evidence bundle:
  `phrase-browser-packages/{checks.py,browser-results.json,commands.jsonl}`,
  six CSVs, twelve HTML/PNG/DOM captures, and `source-copy` containing the exact
  matching inputs/reviews. Reference CSS SHA-256 and registry binding are recorded
  in that lane's report. These are real exported payloads, not native Anki tests.

## Implementation Notes

- Deduplicated against completed T-067 (automatic morphology appearance), T-086
  (phrase recipes), T-088 (phrase export/layout binding) and T-089 (integration).
  None separately tracks the missing application of presentation settings in
  phrase output. This task is the narrow integration regression, not a reopening
  of their broader work or the pending native acceptance lane.
- `phrase_recipes.render_exercise` uses terminology but emits bare divs;
  `reference_templates.py` scopes theme/appearance rules to morphology markup.
  Keep implementation choices with the repair worker. ROUND1 makes no fixes.
- 2026-10-11T18:41:11Z: verification pass
- 2026-10-11T18:41:19Z: Implementation-only completion after verification 2026-10-11T18:41:11Z: both phrase recipes use shared configured appearance; General/Python/Security review findings fixed and fresh disposition verification clean; final ruff/mypy/1025 pytest passed. Native Desktop/AnkiMobile auto acceptance remains PENDING under the scoped 2026-10-11 maintainer deferral; docs checklist unchanged.
