---
id: T-097-keep-principal-part-context-readable-in-native
title: Keep principal-part context readable in native Desktop night mode
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#automatic-lightdark-appearance
dependencies: []
updated_at: "2026-10-13T22:50:18Z"
---

# T-097-keep-principal-part-context-readable-in-native Keep principal-part context readable in native Desktop night mode

## Description

ROUND5 native acceptance on 2026-10-13 confirmed that principal-part answer
context becomes barely readable in Linux Anki Desktop night mode. Tested source
is f8cd5dc85df946aa9a66ac5a74d6edab07cfff64 with reference CSS v3,
aqt/anki 26.9.3, Python 3.14.2, and a real Qt reviewer after native CSV import.

Reproduce with the exact reference note type/CSS and a synthetic principal-part
completion or recognition card. Reveal its answer, then use native Preferences
to change Light to Dark and back to Light without reimport. The morphology panel
correctly honors fixed/default/auto appearance, but the sibling
`.latinitas-context` has fixed foreground rgb(68,68,68) on a transparent background.
Native Anki changes the body background from rgb(255,255,255) to rgb(44,44,44).
Context contrast falls from 9.74:1 to 1.43:1. Actual Qt captures show the repeated
principal forms/meaning almost disappearing in night mode. This is outside the
fixed morphology panel, so fixed light does not protect it.

Deduplication: T-067 implemented auto panel colors and deferred full native
acceptance; T-096 narrowed the release gate to Linux. Neither records or fixes
this surrounding-context contrast defect. Do not misclassify correct fixed-panel
behavior as an auto recoloring failure.

## Acceptance

- Principal-part context remains readable in actual Desktop light/night/light
  with both palettes and default/light/dark/auto panels, for both recipes.
- Preserve fixed panel colors and automatic panel recoloring, identity, slots,
  and approved retained-update boundaries. No scripted theme/class injection.
- Repeat native import/reviewer/Preferences evidence on isolated synthetic data;
  inspect the full answer and disclosure, not only the morphology panel.
- Assess related fixed-color personal text against native night backgrounds,
  without overwriting Personal Notes or claiming an untested defect there.

## Verification Notes

- Confirmed in ROUND5 on 2026-10-13. Native muted/default controls have body
  classes `card card1 isLin fancy night_mode nightMode` in Dark, with unchanged
  context color; return-to-Light restores readability. Monochrome/auto Qt
  captures independently show the same problem. Evidence remains local in the
  ROUND5 acceptance bundle for parent review; no fix or release performed.

## Implementation Notes

This is a native readability finding, not a failure of generated claim review,
CSS installation, approved payload import, or Anki's scheduling behavior.

- Implemented reference style v4 as a manual style update: only shared context
  and plain Personal Notes gain client-night foregrounds. Panels, templates,
  identities, slots, schemas, personal content and update APIs are unchanged.
- Native Anki 26.9.3 / Python 3.14.2, isolated synthetic CSV imports: both
  palettes, default/light/dark/auto and both recipes passed actual Preferences
  light/night/light (48 states). Full context, notes and disclosures inspected.
  Context night contrast improved 1.43 to 10.28; plain Personal Notes 2.43 to
  7.27. Light contrasts remain 9.74 and 5.74. Native captures and pointer clip
  were inspected; browser class simulation is supplemental only.
- Real closed v3 collection capture refused with manual-setup/no-rebinding
  guidance; fresh v4 capture passed. CSS SHA-256 is
  `18cd5b7d2e5b50f90ac017db225d49f7ff11fa57a35058487300e4b56457945e`.
  No automatic installation, migration or reconciliation bypass was added.
- Workflow-v3: rendered RED at 1.4338 contrast, GREEN; dedicated simplifier
  made no edits. Separate General/Python/Database lanes found no concrete
  issues. Fresh candidate validation found DOC-1: "The unreleased changelog
  points readers to CSS v3 for current features even though the current shared
  styling is v4." Fixed both current-feature references after failing/passing
  assertions; fresh disposition verification resolved DOC-1, no new findings.
- Final Ruff, mypy (107 source files), pytest (1046 passed), format and diff
  checks passed. Parent final high native acceptance after T097–T099 remains
  pending; AnkiMobile remains unverified and excluded. No release performed.
- 2026-10-13T22:50:12Z: verification pass
- 2026-10-13T22:50:18Z: Verified 2026-10-13T22:50:12Z: v4 native readability all48states; reviewed and final gates passed. Parent high native repeat after T097-T099 remains pending; Mobile unverified.
