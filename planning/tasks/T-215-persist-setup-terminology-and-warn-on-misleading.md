---
id: T-215-persist-setup-terminology-and-warn-on-misleading
title: Persist setup terminology and warn on misleading profile suffixes
status: todo
priority: low
spec_ref: specs/v0.5.1.md#profile-setup-clarity
dependencies:
    - T-214-expose-reference-setup-and-compare-known-css
updated_at: "2026-11-05T22:11:24Z"
---

# T-215-persist-setup-terminology-and-warn-on-misleading Persist setup terminology and warn on misleading profile suffixes

## Description

Deliver FR-7/FR-8 setup/profile clarity using existing versioned terminology and
JSON serialization, not a new storage format.

## Acceptance

- setup --terminology user|latin persists the setting; reconfigure preserves it
  unless explicit. Preview/generate flags override it, and preview identifies
  profile versus override even when both values match.
- German saved-user generation equals equivalent explicit-user override bytes.
  Check Latin override, unsupported user language and no schema downgrade.
- Saving .toml or another misleading suffix warns before writing that content is
  JSON and recommends .json. Existing JSON-in-.toml loads; .json/default does not warn.
- Update examples/docs to state JSON and use .json paths; never claim TOML support.

## Verification Notes

Exercise public setup/preview/generate persistence/override, suffix and legacy-loading
scenarios. Run mandatory ruff → mypy → pytest chain; follow full spec compatibility,
evidence and permission requirements.

## Implementation Notes
