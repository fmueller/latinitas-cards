---
id: T-151-allow-routine-dependency-updates-without-freezing
title: Allow routine dependency updates without freezing audit snapshots
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-24T08:03:02Z"
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
- PRs #73 (certifi), #74 (publishing action), and #75 (ruff) were rebased onto
  successive main heads and rebase-merged in that order, each with nine passing
  checks and no outstanding review findings.
- Main Build passed after every merge: runs 40074788436, 40075350392, and
  40075947365; CodeQL passed too. No publication workflow was triggered.
- The combined dependency set passed the full mandatory chain again: 2643
  passed, 7 skipped. Taskrail verification passed at 2026-10-24T08:03:02Z.
- GitHub still reports alert 145 open with the old certifi requirement, although
  main locks its first patched version, 2026.10.19. No alert was dismissed.

## Implementation Notes

Keep schema versions, canonical digest vectors, security floors, supported Python
versions, and publication guards: these describe contracts rather than incidental
dependency inventory. Historical audit content remains unchanged; git preserves
its history without a test pinning its bytes.
- 2026-10-24T08:03:02Z: verification pass
