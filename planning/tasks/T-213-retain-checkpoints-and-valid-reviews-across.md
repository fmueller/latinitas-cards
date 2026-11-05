---
id: T-213-retain-checkpoints-and-valid-reviews-across
title: Retain checkpoints and valid reviews across package renames
status: todo
priority: medium
spec_ref: specs/v0.5.1.md#stable-checkpoint-location
dependencies:
    - T-212-hide-withheld-morphology-through-persisted
updated_at: "2026-11-05T22:11:24Z"
---

# T-213-retain-checkpoints-and-valid-reviews-across Retain checkpoints and valid reviews across package renames

## Description

Deliver FR-5 stable checkpoint lookup/migration and valid review continuity through
approved source-stream authority and v0.5.0 adapters. Own export-state consumers.

## Acceptance

- Apply approved stable default, --checkpoint override/precedence, separate from CSV
  identity state. Errors name the expected path and recovery, not presumed schedule loss.
- Export A then same-stream B with a new filename finds prior-export evidence without
  fresh-import approval. Reviewed PPP and another unchanged claim retain decisions/
  eligibility; changed content invalidates affected decisions only.
- Different/uncertain bindings, including overlapping note GUIDs, cannot inherit
  state. Profile pathname alone is insufficient authority.
- Explicit legacy migration validates binding/data, preserves old files until success,
  detects competing checkpoints and handles interruption without silent merging or
  overwriting. Cover missing/corrupt/stale state and explicit override.
- Preserve identities/evidence. Checkpoints prove neither destination import nor
  scheduling continuity and grant no Anki lifecycle authority.

## Verification Notes

Exercise public preview/generate rename/re-export, reviewed-state reuse and negative/
recovery cases; inspect exact state and generated identities. Run mandatory ruff →
mypy → pytest chain; follow full spec compatibility/evidence/permission requirements.

## Implementation Notes
