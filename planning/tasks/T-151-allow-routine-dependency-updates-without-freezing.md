---
id: T-151-allow-routine-dependency-updates-without-freezing
title: Allow routine dependency updates without freezing audit snapshots
status: in_progress
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-24T07:08:30Z"
---

# T-151-allow-routine-dependency-updates-without-freezing Allow routine dependency updates without freezing audit snapshots

## Description

Keep the install and release safety contracts in specs/v0.3.0.md#pypi-publishing
without treating a dated dependency audit as an immutable current lockfile.

## Acceptance

- Routine dependency version updates do not fail tests solely because the lock
  inventory or audit snapshot changed.
- Keep optional NLP isolation, the fsspec advisory boundary, package install
  checks, and release workflow safety coverage.
- Clarify that the license audit records reviewed versions, not a live inventory.
- Rebase and merge Dependabot PRs sequentially only after green CI, and verify
  Build on main after each merge.

## Verification Notes

- Reproduced both original failures with the actual PR #75 lock patch; the same
  six focused tests pass after the change with either PR #75 or PR #73 applied.
- Mandatory chain passed: ruff, mypy (171 source files), pytest (2643 passed,
  7 skipped, one existing Pydantic serializer warning). Format check passed.
- Simplifier made no changes. Independent review retained all install/security
  invariants; a contradictory audit-refresh instruction was corrected and rechecked.
- PyPI metadata for certifi 2026.10.19 and ruff 0.17.1 retains the audited
  MPL-2.0 and MIT licenses respectively; both require Python >=3.7.
- Sequential PR and main CI gates pending delivery.

## Implementation Notes

Keep schema versions, canonical digest vectors, security floors, supported Python
versions, and publication guards: these describe contracts rather than incidental
dependency inventory. Historical audit content remains unchanged; git preserves
its history without a test pinning its bytes.
