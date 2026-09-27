---
id: T-031-require-evidence-backed-field-suggestions-and
title: Require evidence-backed field suggestions and ambiguity confirmation
status: todo
priority: medium
spec_ref: specs/v0.1.0.md#mapping-evidence-and-confirmation
dependencies:
    - T-012-build-assisted-profile-setup
updated_at: "2026-09-27T09:36:15Z"
---

# T-031-require-evidence-backed-field-suggestions-and Require evidence-backed field suggestions and ambiguity confirmation

## Description

Improve profile_setup.py and setup_flow.py suggestion evidence and confirmation without
hard-coding user field names.

## Acceptance

- Synthetic LatinumB_Lektion lesson values must not outrank Latein Latin entries on name
  similarity alone. Show representative candidate values and suggestion reasons.
- Equivalent renamed fixtures behave consistently. Ties, sparse/conflicting evidence,
  and misleading names require explicit choice before a usable profile can be saved.
- Strong suggestions still need confirmation; retain manual overrides, deterministic
  saved mappings, explicit semantic role/order choice, and reload behavior.
- Do not treat success with an already confirmed mapping as ranking-improvement evidence.

## Verification Notes

- Use asymmetric fixtures for ranking, renamed fields, ties, rejected confirmation,
  override, and profile reload. Run the mandatory ruff/mypy/pytest chain.
- Record evidence when executed; this task remains unimplemented.

## Implementation Notes
- Keep this independent of new recipe scope and automated grammatical guessing.
- Medium priority lets interruption recovery (T-036) and scoped identity (T-029) run
  first without inventing a technical dependency. T-032 then supplies normalized
  eligibility before T-030; mapping remains a mandatory v0.1.0 release gate.
