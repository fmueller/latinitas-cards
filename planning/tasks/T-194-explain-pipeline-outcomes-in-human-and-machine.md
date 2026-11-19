---
id: T-194-explain-pipeline-outcomes-in-human-and-machine
title: Explain pipeline outcomes in human and machine reports
status: completed
priority: medium
spec_ref: specs/v0.5.0.md#outcome-reports-and-anki-acceptance-metadata
dependencies:
    - T-193-orchestrate-inspectable-stages-with-bounded-resume
updated_at: "2026-11-19T01:49:15Z"
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

- 2026-11-18T23:25:56Z: High-risk review round 2/2 exhausted: stage-only report restoration validates valid human unanalysed.json authority as automatic analysis, inventing analysis_processing_failed and incomplete status. Public CLI resume then stage select reproduced it; required correction and regression remain. Candidate preserved uncommitted; no completion or push.
- 2026-11-19T01:30:51Z: Maintainer standing exception permits additional confirmed in-scope correction rounds; resume preserved candidate for human-authority applicability regression and correction. Historical rounds 1 and 2 remain consumed; this is round 3.
- 2026-11-19T01:49:08Z: verification pass
- 2026-11-19T01:49:15Z: Implemented readable yield and strict versioned reports over current stages, selection and validated receipts. Verification passed at 2026-11-19T01:49:08Z after final Ruff/mypy/pytest gate (3648 passed,7 skipped,1 warning), public manual campaign and independent high-risk review. Round3 under maintainer standing exception resolved human-authority stage-only applicability; historical rounds and blocker evidence retained. Production policy disabled, missing token bound deferred; no inference/judges/native/release acceptance. Outer loop owns integrated acceptance and successor selection.
