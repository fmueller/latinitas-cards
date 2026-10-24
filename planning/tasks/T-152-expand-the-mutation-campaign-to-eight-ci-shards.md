---
id: T-152-expand-the-mutation-campaign-to-eight-ci-shards
title: Expand the mutation campaign to eight CI shards
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies: []
updated_at: "2026-10-24T07:36:02Z"
---

# T-152-expand-the-mutation-campaign-to-eight-ci-shards Expand the mutation campaign to eight CI shards

## Description

The four-shard campaign completed 98.1% of mutants, but shards 1 and 3 hit
the 120-minute limit. Increase parallelism without changing mutation scope.

## Acceptance

- Run eight disjoint shards (0 through 7), keeping the source assignment algorithm,
  120-minute limit, full test collection and fail-closed completeness checks.
- Update scope documentation and guards and exercise uneven eight-shard tails.
- Validate and review, commit/push main, dispatch mutation CI and monitor it.

## Verification Notes

- Red: updated configuration guard rejected the original four-shard workflow.
  Green: six focused tests, configuration guard and result/runner suites pass.
- Actual inventory: 84 files assigned exactly once, four shards of 11 files
  and four of 10. The implementation is unchanged; tests cover both four and
  eight shards, including initializers and uneven tails.
- Dedicated code-simplifier made no edits. Full mandatory chain passed:
  Ruff, mypy (171 files), pytest (2,644 passed, 7 skipped, one existing
  serializer warning, 368.02 seconds). Formatting and Taskrail validation pass.
- Independent General (code-review), test-coverage and separate claims reviews
  found no defects. Fresh overall/disposition verifier confirmed the clean
  outcome and reran focused tests and mutation shell suites successfully.
  No findings required fixes or deferrals. No application, authentication,
  data or shard implementation changes required specialist security review.
- The next full mutation campaign remains unverified until CI completes.

## Implementation Notes

- User authorizes eight shards, committing/pushing main and rerunning CI.
- Synced main's dependency-test updates without modifying that work. T-151 is
  active there, so Taskrail refused to start this task; its state is preserved.
  This task remains queued unless its normal lifecycle becomes available.
- 2026-10-24T07:36:02Z: verification pass
