---
id: T-198-present-acceptance-evidence-and-morphology-without
title: Present acceptance evidence and morphology without front leakage
status: completed
priority: medium
spec_ref: specs/v0.5.0.md#semantic-color-layers
dependencies:
    - T-197-export-curated-application-cards-only-after
updated_at: "2026-11-19T09:13:30Z"
---

# T-198-present-acceptance-evidence-and-morphology-without Present acceptance evidence and morphology without front leakage

## Description

P50-5 rendering/themes and opt-in layout migration, also covering Outcome Reports And
Anki Acceptance Metadata. Consume recipe/report data; own shared answer presentation.
Follow approved ownership/curated-approval contracts and spec execution/validation rules.

## Acceptance

- Answers show small human/automatic status, keyboard-accessible collapsible evidence
  (German: Warum diese Antwort?), concise check basis and limitations. Mixed siblings
  retain per-card status; no uncalibrated confidence percentages on cards/tags.
- Namespaced note tags conservatively summarize origin/source, never make a partly
  reviewed note wholly human-approved. Preserve user/source tags and field ownership.
- Morphology themes support course conventions, light/dark and non-color cues.
  Analysis defaults to answers; neutral front highlighting cannot reveal tested
  analysis. No v0.5.5 syntax/construction colors.
- Theme changes preserve linguistic acceptance/data/identity but obey curated rendered
  reapproval. No silent installed-template replacement; opt-in migration preserves keys
  and scheduling.

## Verification Notes

Render/inspect German/English light/dark, collapsed/expanded, mixed-status and opted-in
recipe states. Verify DOM/keyboard access and record interaction evidence. Check ownership,
identity invariance and curated invalidation. Native acceptance remains T-201. Run the
mandatory chain.

## Implementation Notes

- 2026-11-19T09:12:59Z: verification pass
- Added opt-in acceptance-1 answer panels, conservative sibling tags and exact
  managed ownership. Family-specific migration proposals preserve legacy field
  prefixes and template ordinals; explicit local confirmation is not installation.
- Curated approval binds the exact prospective learning answer; unapproved views
  remain neutral and label hypothetical appearance. Theme changes require rendered
  reapproval without changing linguistic decisions or semantic identities.
- Final frozen ruff, mypy and pytest chain passed: 3774 passed, 7 intentional
  alias-matrix skips, one existing serializer warning. High-risk independent review,
  scoped simplification/fix verification and final outcome reconciliation closed
  without unresolved blockers. Earlier failed/superseded checks remain recorded.
- Original synthetic engineering fixtures exercised German/English light/dark,
  collapsed/expanded and mixed-origin cards; keyboard/DOM checks, inspected captures,
  a real pointer/keyboard interaction clip and 20 attestation checks passed.
- No real content, linguistic or empirical approval, provider inference, or native
  Anki scheduling/history preservation is claimed. T-200 approval and T-201 native
  acceptance, including T-195 resolution/history uncertainty, remain external gates.
- 2026-11-19T09:13:30Z: Offline engineering implementation verified at 2026-11-19T09:12:59Z; native Anki, real content approval and T-200/T-201 gates remain pending.
