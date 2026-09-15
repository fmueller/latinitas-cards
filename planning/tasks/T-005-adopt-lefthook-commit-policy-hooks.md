---
id: T-005-adopt-lefthook-commit-policy-hooks
title: Adopt lefthook commit policy hooks
status: completed
priority: high
spec_ref: specs/v0.1.0.md#commit-policy-hooks
dependencies: []
updated_at: "2026-09-15T07:38:37Z"
---

# T-005-adopt-lefthook-commit-policy-hooks Adopt lefthook commit policy hooks

## Description

Replace the `pre-commit` framework with the lefthook + mise hook setup used in merge-carlo
and orgtop, adapted to Poetry, as described in `specs/v0.1.0.md#commit-policy-hooks`.

## Acceptance

- `lefthook.yml` defines `pre-commit`, `commit-msg`, and `pre-push` hooks; `.pre-commit-config.yaml`
  and the `pre-commit` dev dependency are gone.
- `mise.toml` pins lefthook and taskrail and provides `setup` and `check` tasks.
- The guard suites under `scripts/*-test.sh` pass locally and run in the CI `Build` workflow.
- AGENTS.md and README describe the setup and the commit message policy.

## Verification Notes

- Verify run 2026-09-15T07:38:37Z passed: ruff check, ruff format --check, mypy, pytest,
  and the commit message, push message, and author guard suites.

## Implementation Notes

- 2026-09-15T07:38:37Z: verification pass
