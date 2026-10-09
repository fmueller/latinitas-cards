---
id: T-065-configure-grammatical-terminology-language
title: Configure grammatical terminology language on generated cards
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#configurable-grammatical-terminology
dependencies:
    - T-068-publish-v0-2-0
    - T-064-record-reviewed-claim-decisions-from-the-cli-for
updated_at: "2026-10-06T18:22:04Z"
---

# T-065-configure-grammatical-terminology-language Configure grammatical terminology language on generated cards

## Description

Add a versioned profile setting for grammatical terminology: Latin (default) or the
profile's user language (German now, English later), following
`specs/v0.2.1.md#configurable-grammatical-terminology`. Today form-parsing labels are
hard-coded Latin (`form_parsing.py` `_LABELS`) and principal-part role labels German
(`cards.py` `_ROLE_DISPLAY_LABELS`). Raised during T-062/T-060 native acceptance.

## Acceptance

- Profile schema carries the terminology setting with Latin default; legacy profiles
  upgrade explicitly; unknown values or unsupported languages fail validation; preview
  reports the effective setting.
- Feature names, controlled-vocabulary feature values and principal-part role labels
  render in Latin or German for both principal-part comparison and form-parsing cards.
- Reviewed claims store canonical values; values outside the vocabulary need review and
  are not silently translated.
- Switching terminology keeps claim fingerprints, review status, note/card identities
  and slots unchanged; prompts stay Latin-first and prose stays in the user language.
- Report effective overrides; changing existing generated answer payloads requires
  renewed managed approval, not renewed linguistic review for display mapping alone.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- T-064 supplies canonical reviewed claims. This task owns shared terminology
  mapping plus principal-part/form-parsing adapters; T-086 consumes it for phrases.
- T-088 owns safe retained-slot managed payload application. Until that extension
  lands, report unsupported operations rather than implying approval enables emission.
