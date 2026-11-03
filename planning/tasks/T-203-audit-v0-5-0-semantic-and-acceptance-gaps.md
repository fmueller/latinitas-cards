---
id: T-203-audit-v0-5-0-semantic-and-acceptance-gaps
title: Audit v0.5.0 semantic and acceptance gaps
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-202-audit-v0-5-0-contract-and-implementation-drift
updated_at: "2026-11-02T23:10:11Z"
---

# T-203-audit-v0-5-0-semantic-and-acceptance-gaps Audit v0.5.0 semantic and acceptance gaps

## Description

Final gap checkpoint after drift resolution. Own bidirectional acceptance-to-delivery-
to-executed-evidence mapping, not a heading coverage percentage. Follow spec isolation
and final-check requirements.

## Acceptance

- Map every acceptance clause, contractual goal, compatibility condition and exclusion
  to tasks, positive/negative scenarios, independently expected results, owners and exact
  evidence revisions. Reverse mapping must expose unjustified tasks.
- Include library-only/LLM/curated journeys, every outcome/deferred resume, default
  more-than-20/batches, blind input, stale/objection/unsupported assertions, native
  presentation and installed invalidation/withdrawal.
- Missing judgments, resources, permission, native evidence, sufficient samples or
  not-evaluable results are gaps/blockers, not mocked passes. Consequential gaps need
  closure or approved scope/spec changes and refreshed review/evidence.
- Inspect actual task bodies/dependencies and taskrail validate/coverage. Coverage
  describes the active spec; inspect inactive scope separately without activation.
  Status/structural coverage cannot prove acceptance.

## Verification Notes

Replay representative positive/negative evidence and compare matrix against full spec.
No open release-impacting gap passes; changes require refreshed affected evaluation,
integration/drift evidence and this audit. Mandatory chain follows code fixes.

## Implementation Notes
