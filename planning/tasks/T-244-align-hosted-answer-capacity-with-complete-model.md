---
id: T-244-align-hosted-answer-capacity-with-complete-model
title: Align hosted answer capacity with complete model task contracts
status: todo
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-243-verify-real-trial-latin-errors-remain-withheld
updated_at: "2026-11-27T07:01:45Z"
---

# T-244-align-hosted-answer-capacity-with-complete-model Align hosted answer capacity with complete model task contracts

## Description

The fresh ten-entry MiniMax replay on 46f1a716 exposed a live integration limit that local programmed-solver QA did not: eight proposals stopped with finish_reason=length at exactly 2048 completion tokens, including one with all tokens spent on reasoning and no answer. Processing has no explicit output-token setting and Limits forbids any value above 2048, although the complete whole-card contract repeats large source/anchor structures. Preserve that failure evidence and make output capacity explicitly configurable and accountable rather than silently increasing defaults or trimming assertions.

## Acceptance

- Add the smallest bounded profile/provider mechanism for explicitly selecting a larger completion allowance for development tasks. Keep the existing 2048 default, existing total-call/token/time/campaign ceilings and unchanged-profile behavior. Do not silently widen qualified runtime support or infer hosted success from offline fixtures.
- Bind the selected allowance into exact outbound requests, run/stage identity, permissions, preflight disclosures and reservations before any call. Test discriminating sufficient/insufficient total budget and changed-cap permission/reuse denial, preserving historical receipts and charges.
- Demonstrate an independently constructed complete answer that exceeds the old allowance through representative public local transport/pipeline controls. Retain every required whole-card assertion and blind independence; do not make proposals pass by dropping source/rendered claims, abbreviating source identity, copying expected answers or repairing malformed responses.
- Keep schema-invalid, truncated and semantically unsupported responses fail-closed. Bonus selected-sense misuse and domina wrong-recipe/segmentation output are model-contract errors, not permission to weaken validation or accept agreement. Do not claim a higher allowance fixes them.
- Document the explicit profile setting, cost/qualification limits, and larger contract response requirements. Prefer concise task-specific presentation instructions where existing shared templates can be selected without changing their ownership; no speculative prompt framework or response protocol rewrite is required.
- Follow implementing-changes, representative regression coverage and final Ruff -> mypy -> full pytest. Parent owns fresh hosted permissions, further dispatch/accounting and report regeneration after exact-candidate acceptance; no live calls or private trial access belong to this task.

## Verification Notes

- Fresh replay: eleven calls, eight length-truncated proposals, one stopped schema-invalid proposal and one complete two-call pair. No retry, nine checker deferrals, all ten cards withheld. Requests were 12238-12941 bytes; two valid responses supplied all four IDs but remained semantically unresolved.
- Updated private-scope report: https://ampcode.com/artifacts/ten-entry-latin-trial-408e5fb9-10c2-4fde-a753-08f91ae7f770 . Raw private files remain with parent; reproduce contract size offline with synthetic data, not the collection.
- Existing public source owners: providers.Limits.max_output_tokens defaults to and caps at 2048; pipeline_profile.Processing has no max_output_tokens field; pipeline_orchestration constructs Limits from intersecting fields. Estimated-runtime qualification explicitly pins 2048 and must not be broadened without qualification.

## Implementation Notes
