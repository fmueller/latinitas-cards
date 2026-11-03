---
id: T-200-evaluate-held-out-automatic-acceptance-against
title: Evaluate held-out automatic acceptance against independent judgments
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#domain-specific-confidence-calibration
dependencies:
    - T-199-document-v0-5-0-workflows-and-reconcile-cli
updated_at: "2026-11-02T23:10:11Z"
---

# T-200-evaluate-held-out-automatic-acceptance-against Evaluate held-out automatic acceptance against independent judgments

## Description

P50-7 evaluator owns held-out evidence under the approved protocol/policy. Requires
permitted data and independent reference judgments; missing access blocks acceptance.
Follow the spec's execution-isolation and final-check requirements.

## Acceptance

- Run frozen held-out samples without tuning on answers. Record exact code/data/
  policy/analyzer/model/checker revisions, strata, counts, adjudication and denominators
  for errors, precision, coverage, withholdings and failures. Include counterexamples
  and classical/biblical/varied materials.
- Compare measurements to approved sample sizes/thresholds. Model agreement/confidence
  is not reference truth. Zero acceptance is not evaluable; inadequate samples or
  missing independent judgments cannot pass.
- Report coverage limits and failed/inconclusive strata. Block affected automatic paths;
  narrowing/disabling requires maintainer disposition and refreshed affected review/
  evidence, never silent threshold adjustment or claiming original scope passed.
- Preserve human/automatic distinction, reliability over yield and no uncertain-card
  learning deck to inflate counts.

## Verification Notes

Independently inspect split/provenance, adjudication and calculations. Repeat a measured
slice and demonstrate zero-acceptance handling. Readiness requires approved criteria or
an explicitly revised approved scope, not mocked results.

## Implementation Notes
