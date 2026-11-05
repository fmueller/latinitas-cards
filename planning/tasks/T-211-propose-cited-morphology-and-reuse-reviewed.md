---
id: T-211-propose-cited-morphology-and-reuse-reviewed
title: Propose cited morphology and reuse reviewed calibrations
status: todo
priority: high
spec_ref: specs/v0.5.1.md#evidence-proposals-and-reviewed-calibration
dependencies:
    - T-210-round-trip-individually-attested-claim-review
    - T-188-check-mapped-deck-evidence-with-latin-rules-and
    - T-190-gate-complete-card-assertions-on-accepted-evidence
    - T-195-track-export-receipts-and-installed-card
updated_at: "2026-11-05T22:11:24Z"
---

# T-211-propose-cited-morphology-and-reuse-reviewed Propose cited morphology and reuse reviewed calibrations

## Description

Deliver FR-2 evidence proposals and bounded reviewer-authorized calibration through
v0.5.0 rules, decisions, whole-assertion gates and receipts/handoff. No second store.

## Acceptance

- Propose segmentation/explanation/form translation with citable source locator,
  version/checksum, value and applicability binding. Missing coverage stays
  unsupported; proposals never imply acceptance.
- A supported regular a-conjugation fixture gains label/segmentation/explanation
  acceptance via recorded calibration without entering every claim value. Record
  reviewer/sample, scope/claim classes, reference/analyzer/policy versions and
  the exact calibration link for every reused decision.
- Irregular/out-of-scope/contradictory/reference-missing cases stay withheld.
  Translations need their own evidence. No acceptance in this route without an
  individual decision or explicit calibration; no blanket lexeme attestation.
- Source/reference/pattern/policy changes invalidate affected reuse; claims inspect
  --check fails closed. Withdrawal feeds installed semantic identities and required
  resolution to T-195 handoff, never CSV omission as installed-content repair.
- Execute approved independent held-out evaluation. Failed/not-evaluable evidence
  blocks the outcome, never lowers the standard or forces a winner.

## Verification Notes

Use approved real references and independently judged positive/negative fixtures;
inspect provenance/drift. T-216 owns installed-card withdrawal/resolution. Run
mandatory ruff → mypy → pytest chain and follow all spec compatibility/permission rules.

## Implementation Notes
