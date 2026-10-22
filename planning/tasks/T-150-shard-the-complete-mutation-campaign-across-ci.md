---
id: T-150-shard-the-complete-mutation-campaign-across-ci
title: Shard the complete mutation campaign across CI jobs
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies: []
updated_at: "2026-10-22T06:49:22Z"
---

# T-150-shard-the-complete-mutation-campaign-across-ci Shard the complete mutation campaign across CI jobs

## Description

The complete covered-line campaign times out after 120 minutes with 19,158
of 24,501 mutants checked. Split generation and execution across CI workers
without excluding more application code or weakening CLI-contract verification.

## Acceptance

- Four non-overlapping source-file shards cover every application Python file,
  including newly added files and package initializers.
- Every shard uses the full test suite, unchanged report-only efficacy and
  fail-closed completeness checks, and a 120-minute limit.
- Preserve unique per-shard artifacts and exact scope manifests; failed shards
  do not cancel their siblings. Local full/differential commands stay unchanged.
- Validate, simplify and independently review; commit and push on main, then
  dispatch the mutation workflow and monitor its exact commit.

## Verification Notes

- Red: five CLI regression cases failed because the shard script was missing.
  Green: five passed, covering exact disjoint assignment, retained settings,
  invalid indices/counts and empty shards without rewriting configuration.
- Four real mutmut fixture runs completed with the configured source shards:
  two mutants killed per module, eight total; each unchanged completeness gate
  passed. Other source modules remained importable. The fixture initially
  lacked a package initializer and imported the installed application instead;
  adding the fixture initializer corrected the setup, not application code.
- Actual inventory check: all 84 files appear exactly once, 21 per shard.
- Dedicated code-simplifier Task made no edits; focused tests and shell guard
  passed. Mandatory chain passed: Ruff; mypy (171 files); pytest -v
  (2,643 passed, 7 skipped, existing serializer warning, 387.51 seconds).
- Independent conventional-review coordinator lacked nested tools; the main
  agent relayed General (code-review), Python, test coverage and separate
  claims/comment-audit assignments. General, Python and claims found no
  defects. Dedicated security/database lanes did not apply: no new credentials,
  untrusted inputs, permissions or persisted application data are involved.
- Test reviewer candidate: "[Medium] Workflow failure propagation is not
  regression-tested" at scripts/check-mutation-config-test.sh:38. Its claimed
  gap is that removing the existing outcome checks would not fail the guard.
  Fresh independent validation rejected it as a T-150 finding: "The candidate
  partially holds as a test-coverage gap, not as a workflow bug. I would drop
  it as a T-150 review finding." Both outcome checks predate this diff and
  remain intact; this is an optional pre-existing test-hardening follow-up,
  not a new failure introduced or widened by the shard implementation.
- Fresh candidate/overall-outcome verifier found no implementation defect,
  and reran the five focused tests and shell guard successfully. No required
  fixes or unresolved findings; one review cycle. Formatting, diff whitespace,
  mutation result/runner suites and Taskrail validation passed.
- The full remote campaign remains unverified until the authorized dispatch
  completes; no speedup or green remote outcome is claimed by local checks.

## Implementation Notes

- Generate only assigned source files using mutmut's supported only_mutate
  setting in disposable CI checkouts. Keep source_paths and covered-line policy.
- Sort by descending byte size then path and deal files round-robin. This
  spreads large modules but is not a guarantee of equal runtime.
- User authorizes committing/pushing main and dispatching the mutation job;
  no PR, release or deployment is requested.
- 2026-10-22T06:49:21Z: verification pass
- 2026-10-22T06:49:22Z: Implementation validated and reviewed; remote full campaign result will be reported after authorized dispatch.
