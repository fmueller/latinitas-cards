---
id: T-006-uv-toolchain
title: Migrate from Poetry to uv
status: completed
priority: high
spec_ref: specs/v0.1.0.md#uv-toolchain
dependencies: []
updated_at: "2026-09-15T08:48:28Z"
---

# T-006-uv-toolchain Migrate from Poetry to uv

## Description

Move environment and dependency management from Poetry to uv, matching merge-carlo, as
described in `specs/v0.1.0.md#uv-toolchain`.

## Acceptance

- `pyproject.toml` has no `[tool.poetry]` sections; `poetry.lock` is replaced by `uv.lock`.
- Every package version in `uv.lock` matches the previous `poetry.lock`.
- The built wheel contains the same package files as the Poetry build.
- `mise.toml`, `lefthook.yml`, CI, Dependabot, AGENTS.md, and README use uv commands.
- The unit tests pass on Python 3.10 and 3.12.

## Verification Notes

- Verify run 2026-09-15T08:48:28Z passed: uv.lock matches every poetry.lock version, the wheel
  file list is unchanged, and ruff, format, mypy, pytest (3.10 and 3.12), and the guard
  suites pass.

## Implementation Notes

- 2026-09-15T08:48:28Z: verification pass
