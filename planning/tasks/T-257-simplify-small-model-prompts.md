---
id: T-257-simplify-small-model-prompts
title: Simplify single-card model prompts for smaller models
status: todo
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-254-replay-approved-ten-german-cards-after-post-trial
updated_at: "2026-11-28T13:23:46Z"
---

# T-257-simplify-small-model-prompts Simplify single-card model prompts for smaller models

## Description

Analyze and implement the smallest prompt simplification that addresses the
observed small-model contract failures, using implementing-changes. Keep transport
retry changes owned by T256 and the baseline Go comparison unchanged in T255.

## Acceptance

- Inspect the actual one-card request builders and latest captured failures.
  Domina/amica/bonus put selected_sense on non-translation assertions; amare,
  laudare/magnus use string source.entry where an object is required; via has
  malformed JSON. Determine which instructions are redundant or contradictory.
- Keep fresh single-card prompts and answer-free blind checks. Simplify the
  model-facing contract without deleting required source/sense/reference evidence,
  weakening nested validators, acceptance semantics, privacy or accounting.
- Make required field shapes clear, including selected_sense and nested values.
  Prefer removing redundant material over adding a second competing contract.
  Version changed prompts/requests so old permissions/reuse cannot silently apply.
- Use captured outputs and synthetic/offline tests during implementation; no live
  MiniMax or other provider inference is authorized. Do not claim hosted small-model
  reliability has improved without a separately measured comparison.
- Complete independent review, final repository checks and Taskrail lifecycle;
  commit and push on main when ready. No migration, automatic approval or release.

## Verification Notes

- Show before/after representative request sizes and preserved task/evidence
  contracts, with focused regression checks. Follow the full implementing-changes
  workflow and final ruff/mypy/pytest chain. Subscribe to CI without waiting on it.

## Implementation Notes
