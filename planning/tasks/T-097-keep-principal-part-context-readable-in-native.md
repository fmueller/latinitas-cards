---
id: T-097-keep-principal-part-context-readable-in-native
title: Keep principal-part context readable in native Desktop night mode
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#automatic-lightdark-appearance
dependencies: []
updated_at: "2026-10-13T21:02:09Z"
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
