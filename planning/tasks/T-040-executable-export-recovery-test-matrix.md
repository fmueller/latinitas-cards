---
id: T-040-executable-export-recovery-test-matrix
title: Collect only executable export recovery test cases
status: completed
priority: medium
spec_ref: specs/v0.1.0.md#interrupt-safe-csv-and-manifest-export
dependencies: []
updated_at: "2026-10-03T10:42:56Z"
---

# T-040-executable-export-recovery-test-matrix Collect only executable export recovery test cases

## Description

Clean up interruption recovery test matrices under the interrupt-safe CSV and
manifest export contract. Collect only moves applicable to the prior file state.

## Acceptance

- Preserve all 24 CSV/manifest and 6 checkpoint executable cases, including
  before/after interruption and mixed/absent-file commit cases.
- Remove runtime skips for nonexistent backups without changing fault injection,
  recovery assertions, exporter behavior, changelog, or published release.
- Pass targeted tests, mandatory ruff/mypy/pytest chain, independent review and
  Taskrail validation.

## Verification Notes

- Baseline targeted file: 81 passed, 10 skipped. Temporary collection contract
  saved original executable parameter tuples; red failed with "Collected 40
  cases, but only 30 executable". Green compared exact tuple equality: 30 cases.
- Pair backups: 4 output + 4 manifest; commits: 8 output + 8 manifest.
  Checkpoint backup: 2; commit: 4. Both interruption timings remain unchanged.
- Original-base checks: targeted 81 passed, full suite 417 passed, zero skips;
  ruff passed, mypy passed for 52 source files, format passed for 141 files.
- Integrated remote baseline: targeted 81 passed; full suite 431 passed, zero
  skips. The additional 14 cases belong to the concurrent mutation configuration
  tests. Exact 30-tuple equivalence still passed; ruff passed, mypy passed for
  53 source files, format passed for 144 files, Taskrail validation passed.
- Dedicated code-simplifier made no edits. Independent General and Python lanes
  each reported "No concrete task-relevant findings." Candidate validation and
  fresh disposition verification confirmed no unresolved findings. Security and
  Database lanes omitted because exporter runtime is unchanged.

## Implementation Notes

- Only applicable-case decorators and skip guards changed; bodies unchanged.
- Reserved parent's T-038 via temporary reference, removed it after creation.
  Concurrent remote commits occupied T-038/T-039; fast-forwarded and recreated
  this task as T-040 through Taskrail, preserving both remote tasks and changes.
- 2026-10-03T10:42:56Z: verification pass
