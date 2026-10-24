---
id: T-153-aggregate-mutation-shard-results-in-actions
title: Aggregate mutation shard results in Actions summary
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies: []
updated_at: "2026-10-24T11:11:15Z"
---

# T-153-aggregate-mutation-shard-results-in-actions Aggregate mutation shard results in Actions summary

## Description

Add an aggregate GitHub Actions report for the eight-shard mutation campaign,
supporting the agent-ready validation evidence in
`specs/v0.3.0.md#agent-ready-cli-contract` without changing mutation scope or
introducing an efficacy floor.

## Acceptance

- Publish total outcome counts, a weighted overall mutation score, and a
  per-shard table in the workflow summary after all shards finish.
- Preserve Markdown and JSON overview artifacts for 30 days.
- Withhold the overall score and fail the reporting job for failed campaigns,
  missing or invalid stats, unchecked/interrupted mutants, or overlapping or
  missing source scopes. Preserve partial counts and diagnostics.
- Keep killed plus timeout as the detected numerator and all generated mutants
  as the denominator; distinguish mutation score from line coverage.

## Verification Notes

- The initial regression run failed because the report command did not exist.
- The 14 CLI regression cases pass, including unequal shard sizes, timeout and
  no-tests accounting, missing/malformed stats, invalid counts, incomplete
  execution, missing shards, scope errors, failed job outcomes, and zero-score
  success without an efficacy floor. Valid partial counts remain visible.
- Replayed the eight artifacts from GitHub Actions run 40076518340: 18,840
  killed, 6 timeout, 5,655 survived, 24,501 total; overall score 76.92%.
- Disposable regressions confirmed that the CLI assertions catch an introduced
  efficacy floor and discarded partial counts. The configuration guard rejects
  nine removals covering the matrix dependency, failure-path publication,
  separate artifact directories, and overview artifact identity and retention.

## Implementation Notes

- Download artifacts into separate shard directories to avoid overwriting
  identically named files. The reporting job runs even after shard failures.
- Hosted Actions execution must be checked after delivery.
- 2026-10-24T11:11:15Z: verification pass
- 2026-10-24T11:11:15Z: Local implementation verified with full ruff/mypy/pytest chain (2658 passed, 7 skipped), 14 reporting regressions, artifact replay, and independent review. Hosted overview execution remains unverified; changes are uncommitted and unpushed.
