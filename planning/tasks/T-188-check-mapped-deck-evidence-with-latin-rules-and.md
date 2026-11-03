---
id: T-188-check-mapped-deck-evidence-with-latin-rules-and
title: Check mapped deck evidence with Latin rules and libraries
status: todo
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-187-implement-versioned-automatic-decisions-and-reuse
updated_at: "2026-11-02T23:10:11Z"
---

# T-188-check-mapped-deck-evidence-with-latin-rules-and Check mapped deck evidence with Latin rules and libraries

## Description

P50-2 evidence adapters outside cli.py. Reuse confirmed deck mappings and focused
analysis modules. Own rule/library evidence under the approved check matrix;
follow the spec's execution-isolation and final-validation rules.

## Acceptance

- Bind each used form/meaning to its confirmed mapping; reuse plausible source forms
  without inventing absent forms or repairing source identity through acceptance.
- Store rule/library results, versions, scope, alternatives and contradictions.
  Supported corroboration can support library-only automatic acceptance; unsupported
  paradigms and missing corroboration remain explicit outcomes.
- A source form alone does not approve translation, segmentation or explanation.
  Contradictions need recorded resolution or withholding, never blanket source trust.
- Missing optional analyzers/resources is processing-failed/unavailable evidence,
  not linguistic rejection. Default dev tests must not import CLTK.

## Verification Notes

Check asymmetric government, ambiguous form/sense, unsupported and missing-resource
fixtures through public stage outputs. Actual resource-backed smoke evidence belongs
to integrated acceptance. Run the full mandatory validation chain.

## Implementation Notes
