---
id: T-253-preserve-phrase-context-evidence-without-duplicate
title: Preserve phrase context evidence without duplicate-run crashes
status: todo
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-252-preserve-shared-source-evidence-for-repeated
updated_at: "2026-11-28T02:03:05Z"
---

# T-253-preserve-phrase-context-evidence-without-duplicate Preserve phrase context evidence without duplicate-run crashes

## Description

Correct T250 round 2's installed-acceptance blocker on 6aca005: identical
constructed-exercise-context values in a valid phrase reference produce duplicate
source evidence after T252's binding-loop deduplication. The provider run aborts
before dispatch without a report, including an unrelated clean phrase; offline
execution completes. See pipeline_orchestration._task and T250's notes, under
specs/v0.5.0.md#whole-spec-acceptance-and-final-checks.

## Acceptance

- Repeated identical phrase contexts reach explicit provider/report outcomes.
  In a mixed run, both the affected phrase and a clean sibling are processed.
- Preserve distinct context text, objective semantics and evidence scopes;
  exercise distinct contexts as a positive control, including equal-length text
  where scopes can coincide. Do not weaken strict duplicate/corruption validation
  or remove required evidence to bypass the failure.
- Retain T252's repeated-entry behavior, including distinct senses. Inspect the
  full evidence/scope assembly before ModelTask construction so this correction
  is not limited to another intermediate insertion point.
- Extend existing public-boundary tests with focused red/green regression
  coverage using synthetic German data and a local endpoint. No broad new matrix.
- Return the corrected candidate to T250's independent installed acceptance
  owner; T253 completion alone does not close A1 or unblock release gates.

## Verification Notes

Use implementing-changes. Run affected checks during development, then final
Ruff, mypy and full pytest before Taskrail verify/complete. No hosted inference,
private collection, replay, policy enablement, import or publication. Commit/push
main after local gates without waiting for CI. Cite verification timestamps,
not ignored artifact paths. T250 remains pending independent acceptance.

## Implementation Notes
