---
id: T-245-invalidate-stale-model-stages-when-output-capacity
title: Invalidate stale model stages when output capacity changes
status: completed
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-244-align-hosted-answer-capacity-with-complete-model
updated_at: "2026-11-27T10:38:34Z"
---

# T-245-invalidate-stale-model-stages-when-output-capacity Invalidate stale model stages when output capacity changes

## Description

Independent integrated QA on 6eeab137 reproduced a pipeline resume defect after changing only processing.max_output_tokens. A complete default-2048 proposal/checker pair reaches research would_accept, but changing to 8192 causes prerequisite_artifact_invalid even with fresh exact permission and sufficient budget. The direct ProviderRun boundary correctly handles the same migration. Historical model stages are rebuilt against the current allowance before being recognized as stale. Fix the orchestration context boundary without weakening proof validation.

## Acceptance

- A public integrated regression first completes both roles at 2048, then changes only the allowance to 8192. Stale permission must not dispatch or reuse old model answers; fresh exact permission must allow new proposal/checker calls under unchanged aggregate ceilings. Preserve original receipts and charges. Retain a positive unchanged-profile reuse control and a nondefault-to-nondefault migration control when it exercises a distinct boundary.
- Recognize stale model-stage context before reconstructing old envelopes against current request identity. Preserve omitted/explicit-2048 historical identity, strict artifact/proof corruption detection, blind independence, complete assertions, and fail-closed semantics. Do not catch all artifact failures, drop proofs or discard historical stages/charges to make resume succeed.
- Keep T-244's explicit capacity limits, wire-bound permission, reservations, total ceilings and exact-2048 runtime qualification unchanged. No hosted call, private collection access, policy enablement or release is part of this task.
- Follow implementing-changes with focused red/green coverage, simplification, independent conventional review and disposition verification. Run final Ruff -> mypy -> full pytest before Taskrail verify/complete, then commit/push to main as authorized. Parent owns independent integrated recheck and prospective hosted replay/accounting.

## Verification Notes

- Independent checkpoint: https://ampcode.com/artifacts/t244-acceptance-checkpoint-2ed70d53-7a72-4b3f-8729-1526adef3828 . Exact candidate 6eeab1376175dba40d8d8f7fcd671df7e47c5b4a; all 15 CI checks passed. Other exercised integration lanes passed, but this defect blocks acceptance.
- Reproduction archive is retained by QA thread T-01a21c36-2ba4-715a-acf4-c1da997a0fab at .amp/in/orb-backup/t244-repro.tar.gz, SHA256 5b43e794b43a06766cef2f3ac16b8fd4279b020c76ee0638df7303091b84f079. It contains public synthetic fixtures only. Main probe: t244/upward.py; complete evidence: t244/QA.md. Run probes sequentially with a fresh T244_SESSION; do not concurrently regenerate their shared disposable certificate.
- Ownership path: pipeline_orchestration._prerequisite reconstructs historical model envelopes through _model_envelope with current Limits. Card versions do not change with the allowance, so missing current-cap success is treated as corruption before dispatch. QA also reproduced 8192 -> 8191; restoring 8192 reused original calls without new dispatch.

## Implementation Notes

- Historical model wrappers supply their committed output capacity during proof reconstruction. The prerequisite traversal still validates referenced artifacts, responses, analyses and independent proofs before returning a stale result. Current dispatch continues using current limits and exact permission; ledgers and campaign charges are retained.
- Public regression covers omitted-2048 -> 8192 and 8192 -> 8191, unchanged and restored-capacity reuse, stale/fresh version-2 permission, blind task equality, original receipts/charges, and isolated referenced checker-response corruption. The original implementation failed the stale-permission assertion with prerequisite_artifact_invalid; corrected cases passed. A deliberate stale-checker-validation bypass failed the strengthened custody assertion and was removed.
- Separate simplifier and high-risk conventional-review completed. Five independent review lanes and fresh custody/overall verifiers found one test-isolation gap, fixed by targeting the restored checker's exact historical response instead of corrupting every response. No production-code findings or unresolved scoped blockers remained.
- Public local-HTTPS reproduction now completes the fresh 8192 pair, reaches research would_accept, retains the original two charges and reuses unchanged requests without dispatch. This is synthetic task-local evidence, not independent integrated acceptance or hosted qualification. Parent owns the integrated recheck; maintainer gates remain unchanged.
- Mutation adapter discovery/doctor passed, but its calibrated research-reader scope does not cover orchestration; the module-wide diff campaign exceeds its ten-minute budget estimate. Focused original-bug and checker-custody fault injections provide bounded effectiveness evidence only. AST comment-hygiene was unavailable because uncomment 3.7.0 is not installed.
- 2026-11-27T10:38:24Z: verification pass
- 2026-11-27T10:38:34Z: Implementation verified at 2026-11-27T10:38:24Z after final Ruff -> mypy -> full pytest (4028 passed, 7 skipped, 3295.68s). High-risk independent review and isolated custody regression resolved; parent independent integrated recheck remains pending.
