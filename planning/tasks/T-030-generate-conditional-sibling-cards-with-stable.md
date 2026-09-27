---
id: T-030-generate-conditional-sibling-cards-with-stable
title: Generate conditional sibling cards with stable template slots
status: completed
priority: high
spec_ref: specs/v0.1.0.md#conditional-sibling-cards
dependencies:
    - T-029-model-coherent-learning-objects-with-stable-note
    - T-032-review-extraction-coverage-and-correct
updated_at: "2026-09-27T15:51:27Z"
---

# T-030-generate-conditional-sibling-cards-with-stable Generate conditional sibling cards with stable template slots

## Description

Render both initial principal-part recipes as cards of their shared learning-object
note, using stable semantic recipe/role keys and non-repurposed template ordinals.

## Acceptance

- Before conditional rendering, extend T-029's authoritative schema with exact supported
  semantic keys, slots, per-card required fields and eligibility rules. Exporter and
  templates consume this contract; T-033 documents it rather than inventing another.
  Profile enablement changes eligibility, not note identity or existing slot order.
  Consume T-032's normalized semantic values; do not duplicate its normalization fix.
- Guard each entire front. Missing targets/answers, unsupported nouns, ambiguous roles,
  and explicit omissions cannot create blank or misleading cards; preserve optional
  gloss behavior and distinct PPP/supine roles without shifting omitted positions.
- Report source entries, objects, notes, cards and zero-eligible omissions separately.
  Allow domain zero-card objects without creating blank cards in native CSV imports.
- Compare eligibility with retained prior export evidence. Withhold affected updates on
  eligibility loss/profile disablement; do not clear fronts or delete/reset cards.
  Missing evidence requires explicit fresh-import confirmation or review-only output,
  not a claim of destination awareness or proof of actual prior import.
- Define the versioned checkpoint's source-scope, note-family/schema and template-registry
  binding, object IDs, exported card-key/slot eligibility and export fingerprints. Missing,
  corrupt or incompatible state requires recovery/review or explicit fresh import, never
  an assumed empty prior card set. Retain last safe evidence for withheld/absent objects;
  parser failures and partial exports cannot erase it. Withhold the whole affected note
  row if shared-field changes could invalidate an existing card.
- Preview never advances the checkpoint. Commit it with output and identity state through
  the existing recoverable boundary, preserving prior scope/assignments/eligibility or
  backups on failure. Extend T-036's interruption tests for this state. It records export,
  not observed import; v0.2's successfully applied destination baseline remains separate.
- Shared note ID, Personal Notes and tags coexist with different card IDs/ordinals and
  independent scheduling. Added supported cards preserve surviving keys and note ID.
- Do not implement FSRS, automatic burying-setting changes, provisioning, or live writes.

## Verification Notes

- Add missing-data, profile-add/remove, slot-stability and shared-data regression tests.
  Cover read-only preview, missing/mismatched checkpoints, withheld-row retention, partial
  exports, failed commit/retry and preservation of surviving-card evidence on additions.
  Run the mandatory ruff/mypy/pytest chain; T-035 verifies native sibling/import behavior.
- Record evidence when executed; no implementation is claimed by this planning task.

## Implementation Notes
- Adapt existing generation/rendering boundaries and preserve T-026/T-027/T-028 behavior.
- 2026-09-27T15:51:23Z: verification pass
