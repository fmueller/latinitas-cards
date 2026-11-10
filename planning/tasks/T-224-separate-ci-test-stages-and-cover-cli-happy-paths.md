---
id: T-224-separate-ci-test-stages-and-cover-cli-happy-paths
title: Separate CI test stages and cover CLI happy paths
status: completed
priority: medium
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies: []
updated_at: "2026-11-10T01:54:05Z"
---

# T-224-separate-ci-test-stages-and-cover-cli-happy-paths Separate CI test stages and cover CLI happy paths

## Description

Separate existing Python coverage into unit, integration and CLI end-to-end
stages, and add offline happy-path coverage through the real CLI entry point.

## Acceptance

- CI runs unit, integration and E2E jobs sequentially on Python 3.13 and 3.14,
  without running the same tests in multiple stages or losing existing coverage.
- Build Result fails when any selected stage fails, is cancelled or is skipped;
  documentation-only scope continues to skip application checks.
- Happy-path E2E tests run real CLI subprocesses in temporary directories and
  assert exported content, repeatability and input preservation.
- Repository guidance describes the categories and targeted commands while
  preserving the single final local validation gate.

## Verification Notes

- RED: the new CI contract failed because the unit stage still invoked all
  tests instead of its own directory. GREEN: the updated stage contract passes.
- Separate collections find 1084 unit, 2171 integration and 17 E2E cases. Their
  node-ID sets are disjoint and equal default collection; no old IDs were lost.
- Focused E2E/integration/contract run: 90 passed. CI scope regression suite
  rejects failure, cancellation and unexpected skips for each added stage.
- Simplification and independent general, CI, Python and claims reviews passed;
  a fresh overall challenge and coordinator reconciliation found no blockers.
- Final ordered gate passed: ruff; mypy (200 source files); pytest (3265 passed,
  7 skipped, one existing Pydantic warning, 648.72 seconds). No executable changes
  followed that run. Taskrail verification passed at 2026-11-10T01:53:41Z.
- Changes remain local and uncommitted; hosted CI has not run this follow-up.

## Implementation Notes

- Reclassify mixed modules by their broadest exercised boundary, preserving
  all original test bodies except two moved-path references. Shared source
  guards live at the tests root; golden harness imports use integration paths.
- Unit, integration and E2E stages each retain both supported Python versions;
  each complete matrix gates the next one. Package and policy guards remain
  parallel after lint; the always-running aggregate requires all selected jobs.
- New authored and USFX happy paths launch real CLI subprocesses in temporary
  directories and verify CSV content, repeatability and source preservation.
- Live Anki/browser acceptance is not implied by these offline E2E tests.
- 2026-11-10T01:53:41Z: verification pass
- 2026-11-10T01:54:05Z: Sequential stages and happy-path E2E coverage verified locally; uncommitted, no new hosted CI run.
