---
id: T-067-automatic-light-dark-appearance
title: Follow client night mode with automatic appearance
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#automatic-lightdark-appearance
dependencies:
    - T-068-publish-v0-2-0
updated_at: "2026-10-06T19:58:13Z"
---

# T-067-automatic-light-dark-appearance Follow client night mode with automatic appearance

## Description

Add `appearance: auto` so generated morphology markup follows the Anki/iOS night mode
through reference CSS instead of a fixed light/dark class, following
`specs/v0.2.1.md#automatic-lightdark-appearance`. Requested by the maintainer during
T-062 native acceptance.

## Acceptance

- Profile schema accepts `auto` alongside `light`/`dark`; preview reports the
  effective value; fixed modes are unchanged.
- `auto` markup has no fixed appearance class; versioned reference CSS selects the dark
  palette under the client night-mode classes and the light palette otherwise, for
  muted and monochrome, without JavaScript.
- Claims, note/card identities and slots stay unchanged; applying to an existing deck
  remains an approved update reimport.
- Native Desktop and AnkiMobile checks confirm night-mode toggling recolours `auto`
  cards readably before claiming verification.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- This task can run independently of shared claim decisions. Browser CSS checks
  support implementation but do not substitute for native Desktop/AnkiMobile
  acceptance. Missing native access is a blocker to verification, not a waiver.
- Document the reference-style version bump and manual/carrier installation; CSV
  does not install CSS. Existing fixed appearance modes remain unchanged.
- Provide a native install/toggle checklist recording Desktop/AnkiMobile versions,
  both palettes, default/fixed/auto states and readability. T-088 owns layout/style
  rebinding guidance; old CSS digests require reviewed manual setup/reconciliation,
  not payload approval alone. Never silently rebind scheduled templates.
