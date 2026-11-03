---
id: T-185-design-independently-judged-acceptance-evaluation
title: Design independently judged acceptance evaluation
status: todo
priority: high
spec_ref: specs/v0.5.0.md#domain-specific-confidence-calibration
dependencies:
    - T-184-design-v0-5-0-acceptance-and-pipeline-contracts
updated_at: "2026-11-02T23:10:11Z"
---

# T-185-design-independently-judged-acceptance-evaluation Design independently judged acceptance evaluation

## Description

P50-1 evaluation-design draft using the contract draft's claim classes. Own the
proposed protocol and feasibility evidence; do not approve numeric policy or
implement consumers. Follow the spec's execution-isolation and validation rules.

## Acceptance

- Identify an independent Latin reference-judgment owner, permissible sources and
  access. Include classical/biblical and varied supported inputs, claim types,
  recipes, ambiguous cases and counterexamples without a processing whitelist.
- Define development/held-out separation, adjudication, units/denominators for
  accepted errors, precision, coverage, withheld cases and processing failures,
  plus uncertainty reporting. Zero accepted items is not evaluable.
- Propose justified minimum sample sizes and explicit pass/fail criteria by stratum
  with source or measured pilot evidence. Account for correlated library/model
  errors; agreement is not truth. Do not tune against held-out answers.
- Define handling of insufficient evidence and failed/inconclusive strata, narrower
  coverage or disabled automatic paths, including approval and evidence refresh.
- Limit work to one documented feasibility/protocol pass. Missing independent
  judgments, data permission or justified thresholds blocks the maintainer gate,
  rather than prompting invented evidence or indefinite research.

## Verification Notes

Review provenance, rationale, holdout separation and competing wrong answers with
the reference owner. Maintainer approval must cover this exact protocol revision;
neither dependent implementation nor release readiness is established.

## Implementation Notes
