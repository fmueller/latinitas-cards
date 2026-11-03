---
id: T-194-explain-pipeline-outcomes-in-human-and-machine
title: Explain pipeline outcomes in human and machine reports
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#outcome-reports-and-anki-acceptance-metadata
dependencies:
    - T-193-orchestrate-inspectable-stages-with-bounded-resume
updated_at: "2026-11-02T23:10:11Z"
---

# T-194-explain-pipeline-outcomes-in-human-and-machine Explain pipeline outcomes in human and machine reports

## Description

P50-4 reporting over actual stage/selection results. Own readable and versioned
machine report renderers, source-entry/passage links and representative examples.
Follow approved contracts and spec execution/validation requirements.

## Acceptance

- Distinguish not generated, deferred/not processed, withheld, processing failed,
  accepted not selected and accepted selected. Provider outage is not linguistic rejection.
- Distinguish source work, stage, claim, candidate and card counts, including one failed
  claim affecting several cards. Never invent undiscovered exercise counts.
- Explain limits, suppression, reuse, alternatives, objections, evidence limitations
  and yield without per-candidate review. Expose human/automatic origin and policy version.
- Partial-budget runs/resume retain valid outcomes and distinguish deferred work from
  failures and unselected acceptance. Do not leak secrets or require private reasoning.

## Verification Notes

Use fixtures and CLI integration for all outcome classes and one-claim/multiple-card
counts. Validate schema/source links, partial resume and error redaction. Run the
mandatory validation chain.

## Implementation Notes
