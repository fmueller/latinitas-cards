---
id: T-041-classify-mutated-cancellation-errors-as-test
title: Classify mutated cancellation errors as test failures
status: completed
priority: medium
spec_ref: specs/v0.1.0.md#interrupt-safe-csv-and-manifest-export
dependencies: []
updated_at: "2026-10-03T11:11:00Z"
---

# T-041-classify-mutated-cancellation-errors-as-test Classify mutated cancellation errors as test failures

## Description

The full CI mutation run 37116944401 passed its unmutated baseline and completed
11,224 of 11,225 mutants. The remaining mutant changed the cancellation recovery
condition from `and` to `or`, converting SystemExit to KeyboardInterrupt, which
escaped the expected-exception assertion and interrupted pytest's runner.

## Acceptance

- Preserve the SystemExit cancellation contract and backup/recovery assertions.
- Classify the wrong KeyboardInterrupt as an assertion failure, not interruption.
- Kill the exact interrupted mutant without changing production or gate policy.
- Pass the mandatory validation chain before verification and completion.

## Verification Notes

- Reproduced `latinitas_cards.preview_export.x_write_principal_part_csv__mutmut_180`
  locally: before the test change, interrupted; afterwards, killed.
- Explicit mutated pytest run failed at `assert isinstance(error.value,
  SystemExit)` with AssertionError, while the unmutated focused test passed.
- Mandatory ruff/mypy/pytest passed: 431 passed, 10 skipped.
- Dedicated simplifier made no changes. Independent General and Python reviewer
  lanes and fresh candidate validation returned "No concrete task-relevant
  findings." No dispositions required; no security/database/framework changes.

## Implementation Notes

- Capture SystemExit and KeyboardInterrupt in the expected-exception context,
  then assert SystemExit. Do not accept the wrong exception or weaken recovery
  assertions, and do not relax interrupted-result rejection.
- 2026-10-03T11:10:09Z: verification pass
- 2026-10-03T11:11:00Z: verification pass
