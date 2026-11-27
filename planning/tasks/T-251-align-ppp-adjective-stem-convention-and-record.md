---
id: T-251-align-ppp-adjective-stem-convention-and-record
title: Align PPP adjective stem convention and record contract-violating reconciliations
status: todo
priority: low
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-247-enforce-explicit-per-card-exercise-and-nested
updated_at: "2026-11-27T20:13:00Z"
---

# T-251-align-ppp-adjective-stem-convention-and-record Align PPP adjective stem convention and record contract-violating reconciliations

## Description

From T-247 review: PPP-anchored adjective_declension cards claim bennett-present-stem/1 because the renderer selects the stem convention by entry POS rather than recipe; reconcile of a contract-violating response raises without recording a failed call (same as pre-existing malformed reconcile). Both fail closed today. Keep German goldens and card identities stable or version them explicitly.

## Acceptance

- The follow-up issue described by verification is resolved.
- Verification evidence is updated.

## Verification Notes

- Re-run task-scoped verification after implementing the fix.
