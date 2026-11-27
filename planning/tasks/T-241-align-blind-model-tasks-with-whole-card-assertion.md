---
id: T-241-align-blind-model-tasks-with-whole-card-assertion
title: Align blind model tasks with whole-card assertion and language contracts
status: todo
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies: []
updated_at: "2026-11-27T01:28:37Z"
---

# T-241-align-blind-model-tasks-with-whole-card-assertion Align blind model tasks with whole-card assertion and language contracts

## Description

Fix the request/checker mismatch reproduced in the ten-entry development trial. Production and declension requests expose only a recipe name, while semantic comparison requires the exact whole-card assertion inventory and structured contents. Both blind roles must receive enough source-bound task information to satisfy that contract without seeing the proposal or being handed the expected answer. Explicitly bind explanatory prose to the selected language.

## Acceptance

- Communicate required assertion IDs, kinds, content representations, evidence and selected-sense rules for the affected form recipes, derived from the owning contract rather than a duplicated answer key.
- Require German explanatory prose for German tasks while retaining Latin surfaces and machine identifiers. Keep supported English workflows intact.
- Demonstrate a source-bound positive proposal/counter comparison through the public pipeline, plus a discriminating incomplete/wrong answer that remains unresolved or contradictory. Preserve blind independence, untrusted-data isolation and complete assertion coverage.
- Keep strict readers, disabled acceptance policy, human approvals, protected study and publication gates unchanged. Do not silently repair malformed outputs or trim inconvenient assertions.
- Use representative regression coverage and the required final Ruff, mypy and full pytest chain, simplification and independent review.

## Verification Notes

- Diagnosis: pipeline_orchestration._task versus semantic_comparison.compare_source_answers. Nine saved schema-valid trial pairs independently reproduced incomplete_model_assertion_coverage; this is not a Latin-quality-only failure.
- Source report: https://ampcode.com/artifacts/ten-entry-latin-trial-408e5fb9-10c2-4fde-a753-08f91ae7f770 . Private trial data stays outside Git. No new hosted calls are needed to reproduce the contract gap.

## Implementation Notes
