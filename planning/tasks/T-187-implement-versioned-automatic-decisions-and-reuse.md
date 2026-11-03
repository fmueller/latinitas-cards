---
id: T-187-implement-versioned-automatic-decisions-and-reuse
title: Implement versioned automatic decisions and reuse bindings
status: todo
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-186-approve-v0-5-0-contracts-and-evaluation-policy
updated_at: "2026-11-02T23:10:11Z"
---

# T-187-implement-versioned-automatic-decisions-and-reuse Implement versioned automatic decisions and reuse bindings

## Description

P50-2 decision storage and strict-reader integration. Own focused claim/sidecar
adapters and stage-result bindings, not a parallel review system. Apply the approved
P50-1 contract and the spec's execution-isolation/final-validation requirements.

## Acceptance

- Human and automatic acceptance remain distinct after serialization, reload and
  regeneration. Preserve historical human decisions and reject unsupported versions.
- Bind reusable successes to source content, extraction, task, language, configuration
  and producer/analyzer/prompt/checker/policy versions. Correctness changes invalidate
  affected results without discarding valid unrelated work.
- Persist stage outcomes, alternatives, objections and evidence. Malformed, stale,
  failed or unknown records cannot authorize export or destination writes.
- Strict deck/corpus readers consume only explicitly upgraded/adapted records.
  Preserve old valid human sidecars and unchanged deck/corpus output compatibility.

## Verification Notes

Use public readers and preview/export interfaces for old/new round trips, changed
source/checker/policy/language, malformed/unsupported versions and unchanged reuse.
Run the mandatory ruff, mypy, pytest chain before verify/complete.

## Implementation Notes
