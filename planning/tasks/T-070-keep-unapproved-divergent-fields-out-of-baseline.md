---
id: T-070-keep-unapproved-divergent-fields-out-of-baseline
title: Keep unapproved divergent fields out of the managed baseline
status: completed
priority: high
spec_ref: specs/v0.2.0.md#managed-update-plans
dependencies: []
updated_at: "2026-10-06T20:17:46Z"
---

# T-070-keep-unapproved-divergent-fields-out-of-baseline Keep unapproved divergent fields out of the managed baseline

## Description

Found by the T-069 adversarial review. After a plan where a managed field (e.g. Lemma)
diverged in the destination from both baseline and proposal, approving and observing only a
different field folded the user's divergent value into the anchor, so the next plan offered
to overwrite the user's edit as a supported update; an explicit `keep_destination` decision
was dropped the same way. This breaks the three-way rule in
specs/v0.2.0.md#managed-update-plans. Release-blocking for T-068.

## Acceptance

- Observed advancement only moves the managed baseline for written fields and for fields where destination already equals the proposal; unapproved divergent fields keep the prior baseline value.
- A replan after partial observation still classifies the divergent field as an unsupported conflict, with and without a `keep_destination` decision.
- Retained divergence is not mistaken for backup restoration; restoration detection still compares the observed destination values.
- Regression test in tests/unit/managed_application_test.py; mandatory ruff/mypy/pytest chain passes.

## Verification Notes

- RED: new parametrized regression
  `test_unapproved_divergent_field_stays_a_conflict_after_partial_observation`
  failed (baseline Lemma became the user's edit) before the fix; GREEN after.
- The reviewer's standalone repro scripts (plain and `keep_destination` variants) now
  replan the divergent Lemma as an unsupported conflict.
- Adversarial re-review found no defects; restoration detection still flags a destination
  reverted to the old baseline (asserted in the regression test).
- Mandatory chain: ruff check, mypy, pytest (840 passed).

## Implementation Notes

- Fix implemented during the T-069 review at maintainer request: emit records `baseline_fields`; observe stores them as anchor fields plus `observed_fields` for consistency checks.
- 2026-10-06T20:17:46Z: verification pass
