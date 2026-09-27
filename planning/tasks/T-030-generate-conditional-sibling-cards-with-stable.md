---
id: T-030-generate-conditional-sibling-cards-with-stable
title: Generate conditional sibling cards with stable template slots
status: todo
priority: high
spec_ref: specs/v0.1.0.md#conditional-sibling-cards
dependencies:
    - T-029-model-coherent-learning-objects-with-stable-note
updated_at: "2026-09-27T08:56:30Z"
---

# T-030-generate-conditional-sibling-cards-with-stable Generate conditional sibling cards with stable template slots

## Description

Render both initial principal-part recipes as cards of their shared learning-object
note, using stable semantic recipe/role keys and non-repurposed template ordinals.

## Acceptance

- Define the exact supported slot registry, required fields and eligibility rules.
  Profile enablement changes eligibility, not note identity or existing slot order.
- Guard each entire front. Missing targets/answers, unsupported nouns, ambiguous roles,
  and explicit omissions cannot create blank or misleading cards; preserve optional
  gloss behavior and distinct PPP/supine roles without shifting omitted positions.
- Report source entries, objects, notes, cards and zero-eligible omissions separately.
  Allow domain zero-card objects without creating blank cards in native CSV imports.
- Compare eligibility with retained prior export evidence. Withhold affected updates on
  eligibility loss/profile disablement; do not clear fronts or delete/reset cards.
  Missing evidence requires explicit fresh-import confirmation or review-only output,
  not a claim of destination awareness or proof of actual prior import.
- Shared note ID, Personal Notes and tags coexist with different card IDs/ordinals and
  independent scheduling. Added supported cards preserve surviving keys and note ID.
- Do not implement FSRS, automatic burying-setting changes, provisioning, or live writes.

## Verification Notes

- Add missing-data, profile-add/remove, slot-stability and shared-data regression tests.
  Run the mandatory ruff/mypy/pytest chain; T-035 verifies native sibling/import behavior.
- Record evidence when executed; no implementation is claimed by this planning task.

## Implementation Notes
- Adapt existing generation/rendering boundaries and preserve T-026/T-027/T-028 behavior.
