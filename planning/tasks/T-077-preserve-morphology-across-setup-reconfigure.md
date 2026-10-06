---
id: T-077-preserve-morphology-across-setup-reconfigure
title: Preserve hand-edited morphology settings across setup reconfiguration
status: todo
priority: medium
spec_ref: specs/v0.2.0.md#morphology-themes
dependencies: []
updated_at: "2026-10-06T20:17:05Z"
---

# T-077-preserve-morphology-across-setup-reconfigure Preserve hand-edited morphology settings across setup reconfiguration

## Description

Found by the T-069 new-user walkthrough. `setup --reconfigure --confirm` silently resets a
hand-edited profile `morphology` section to muted/light/static, and a partial morphology object
without `version` is accepted silently. Themes are only selectable by editing profile JSON.
Not release-blocking.

## Acceptance

- Reconfiguration preserves or explicitly reports changes to existing morphology settings.
- Morphology objects without an explicit version fail clearly or are documented as defaulted.
- Regression tests cover both cases.

## Verification Notes

- Pending.

## Implementation Notes


