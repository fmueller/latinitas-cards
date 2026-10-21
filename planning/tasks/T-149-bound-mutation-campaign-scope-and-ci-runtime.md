---
id: T-149-bound-mutation-campaign-scope-and-ci-runtime
title: Bound mutation campaign scope and CI runtime
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies: []
updated_at: "2026-10-21T17:05:26Z"
---

# T-149-bound-mutation-campaign-scope-and-ci-runtime Bound mutation campaign scope and CI runtime

## Description

Reduce mutation campaign work without excluding application contracts, within
the active spec's CLI-contract verification. The latest remote run passed
baseline collection but reached only about 55% of 27,155 mutants in 90 minutes.

## Acceptance

- Measure covered-line mutation generation and document its denominator change.
- Audit low-value text candidates; do not exclude user-facing output contracts.
- Increase the CI job timeout to 120 minutes, preserving completeness checks.
- Pass focused regressions, mandatory validation and independent review.

## Verification Notes

- Red: coverage-config regression failed (1 failed, 21 passed); shell guard
  rejected the original 90-minute timeout. Green: 22 tests and shell guard pass.
- Mandatory chain passed: Ruff; mypy (170 source files); pytest -v
  (2,638 passed, 7 skipped, one existing serializer warning, 639.73 seconds).
- Mutation configuration and result/runner shell suites, formatting and
  Taskrail validation passed.
- Fresh covered-line generation completed: 84 files, 24,509 mutants versus
  the prior remote 27,155 (2,646 fewer, 9.7%). This measures generation, not
  campaign completion or expected wall time. Coverage collection adds setup
  cost; 120-minute remote completion remains unproven.
- Measurement command: `uv run mutmut run --max-children 2
  latinitas_cards.__measurement_only_no_match__`. Generation and full stats
  collection completed, producing mutmut-stats.json. The deliberately unmatched
  selector then exited 1 with "Filtered for specific mutants, but nothing
  matches", as intended to prevent launching a full campaign. This is not a
  successful full mutmut run. Generated measurement files were removed.
- Dedicated personal code-simplifier Task made no changes. General independent
  code-reviewer returned "No concrete task-relevant findings." Fresh candidate
  validation returned the same conclusion. No findings required disposition.
  General covers config/workflow and trivial Python assertions; no product,
  security, database or framework invariants changed to require other lanes.
- Fresh disposition-verification code-reviewer inspected the final diff and
  evidence: "No concrete task-relevant findings." One review cycle, no
  unresolved issues; no full local or remote campaign claimed.

## Implementation Notes

- No logging framework calls were found in the application audit. Console
  output includes CLI diagnostics, tables and warnings; no blanket string or
  console exclusions are justified. No application source changes are needed.
- Renumbered from the local T-148 draft after syncing main, where the README
  publication task had already taken that short key. Release changes preserved.
- 2026-10-21T17:05:26Z: verification pass
- 2026-10-21T17:05:26Z: Preserve release updates; covered-line filtering and 120-minute timeout ready for remote campaign.
