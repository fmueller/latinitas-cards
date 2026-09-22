---
id: T-015-add-mutation-testing-infrastructure
title: Add mutation testing infrastructure
status: completed
priority: medium
spec_ref: specs/v0.1.0.md#non-functional-requirements
dependencies: []
updated_at: "2026-09-22T17:37:36Z"
---

# T-015-add-mutation-testing-infrastructure Add mutation testing infrastructure

## Description

Add a Python mutation-testing lane that measures whether the existing unit tests detect
behavioral changes. Keep expensive full runs out of pull-request CI, provide a differential
developer loop, and collect a clean baseline before adopting an efficacy threshold.

## Acceptance

- mutmut is locked as a development dependency and targets the application package without
  recursively collecting its generated mutant tree.
- A differential command mutates exactly the Python modules changed from a configurable base
  reference, ignores unrelated cached outcomes, and fails on tool errors or incomplete selected
  results.
- A full command and a weekly/manual GitHub Actions workflow run all discovered mutants, preserve
  raw text and machine-readable results, and fail when mutation execution or result collection is
  incomplete.
- The initial policy is report-only for efficacy: it reports stable per-module raw counts without
  copying a threshold from another repository. The documentation explains how a clean baseline
  will be used to select a later floor.
- Regression tests cover result parsing, exact differential scope, stale coverage-map removal,
  and propagation of mutation-tool failures.
- The ordinary repository gate exercises the mutation infrastructure tests without running the
  expensive mutation suite.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- 2026-09-22T17:37:06Z: mandatory `ruff`, strict `mypy`, and verbose `pytest`
  chain passed (46 tests); mutation result/configuration shell suites and Taskrail validation
  passed; independent disposition verification resolved or soundly dispositioned all three
  validated review findings.
- A complete report-only mutation run evaluated 3,573 mutants: 1,514 killed, 1,624 survived,
  435 had no tests, and none were unchecked, interrupted, or unknown.

## Implementation Notes

- Do not add a numerical mutation efficacy floor until a clean full run establishes a project
  baseline.
- Full mutation testing belongs in weekly/manual CI, not the pull-request build.
- 2026-09-22T17:37:30Z: verification pass
