---
id: T-250-verify-integrated-post-trial-fixes-on-current
title: Verify integrated post-trial fixes on current transport
status: todo
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-249-evaluate-bounded-pydantic-ai-migration-feasibility
updated_at: "2026-11-27T16:40:12Z"
---

# T-250-verify-integrated-post-trial-fixes-on-current Verify integrated post-trial fixes on current transport

## Description

A1 independent offline acceptance of post-trial fixes on the frozen installed
current-transport candidate. Requires accepted F1/F2 and recorded M1 outcome, not
M2 or a hosted replay. This owner must differ from F1/F2 implementers. See
docs/v0.5.0-post-trial-plan.md.

## Acceptance

- Derive expectations from the approved spec, then exercise installed public pipeline
  boundaries with synthetic German production/recognition, multi-binding phrases,
  English compatibility, nested/malformed/presentation failures and unsupported Latin
  withheld. Inspect outputs and persisted state, not only unit mocks or green CI.
- Exercise synthetic historical-layout read/resume, strict corruption rejection,
  exact permissions, retained failed/unknown charges, one-card blind independence,
  unchanged-scope reuse and offline no-network behavior. Use representative happy
  paths/important negatives rather than another broad matrix. Never commit private
  collection/store/model-response fixtures.
- Capture exact source/wheel/installed identity, commands, expected/observed results,
  coverage and evidence gaps. Prior task-local checks are not independent acceptance.
  Return confirmed defects to their owner under implementing-changes; rerun affected
  scenarios on the corrected frozen candidate.
- Complete on passed offline acceptance alone, including M1 defer/infeasible outcomes.
  Do not wait for R1 or claim hosted quality, linguistic authority, protected T200
  evaluation, native Anki acceptance or release readiness.
- T201 remains blocked pending accepted A1 and M2 if M1 adopts. R1 stays separately
  permissioned; declined/unperformed replay is not passed. Adopted M2 must independently
  repeat these offline controls on the migrated candidate and satisfy separate
  qualification obligations.

## Verification Notes

Follow implementing-changes for any product fixes and use installed acceptance for
this verification task. Missing material evidence blocks acceptance. Final Ruff →
mypy → full pytest precedes verify/complete. No hosted inference, collection edits,
import or publication. Commit/push evidence/task completion on main without waiting
for CI; parent owns final CI-fix supervision.

## Implementation Notes
