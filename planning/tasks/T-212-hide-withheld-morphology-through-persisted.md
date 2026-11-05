---
id: T-212-hide-withheld-morphology-through-persisted
title: Hide withheld morphology through persisted presentation settings
status: todo
priority: high
spec_ref: specs/v0.5.1.md#withheld-analysis-presentation
dependencies:
    - T-211-propose-cited-morphology-and-reuse-reviewed
    - T-198-present-acceptance-evidence-and-morphology-without
updated_at: "2026-11-05T22:11:24Z"
---

# T-212-hide-withheld-morphology-through-persisted Hide withheld morphology through persisted presentation settings

## Description

Deliver FR-2 presentation-only hide/neutral modes through approved profile and
managed CSS/layout contracts. Preserve the old default.

## Acceptance

- Persist presentation mode. Hide mode contains no withheld placeholders; neutral
  mode shows at most one line. Accepted analysis remains visible in mixed cards.
- Hiding never accepts claims, changes decisions or unlocks cards. Preserve identity
  and retained fields; installed-template changes require explicit migration.
- Render default/hidden/neutral/mixed states in German/English light/dark without
  front leakage; managed plans bind the effective setting, no user CSS edits needed.

## Verification Notes

Inspect rendered screenshots and DOM for all affected states; exercise/record any
disclosure keyboard/pointer behavior. Check regeneration and supported managed-plan
identity/decision invariance. T-216 owns native evidence. Run mandatory ruff → mypy
→ pytest chain and follow full spec compatibility/evidence/permission boundaries.

## Implementation Notes
