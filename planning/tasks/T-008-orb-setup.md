---
id: T-008-orb-setup
title: Prepare cached orb development environments
status: completed
priority: medium
spec_ref: specs/v0.1.0.md#uv-toolchain
dependencies: []
updated_at: "2026-09-20T14:24:14Z"
---

# T-008-orb-setup Prepare cached orb development environments

## Description

Prepare snapshot-friendly orb lifecycle scripts using the pinned development
environment in specs/v0.1.0.md#uv-toolchain.

## Acceptance

- Executable setup installs locked tools, Python dependencies, and hooks.
- Repeated setup converges without duplicate profile configuration or downloads.
- Clean non-interactive login shells resolve the pinned toolchain.
- Resume finishes in seconds without dependency installation.
- Portal metadata is ignored and the repository validation chain passes.

## Verification Notes

- Executable assertion failed before implementation and passed after chmod;
  bash syntax validation passed for both lifecycle scripts.
- Setup took 47.091 seconds with a fresh Python environment, then 0.960
  seconds with no downloads; resume took 0.003 seconds.
- A minimal-environment login shell resolved uv 0.9.17, lefthook 2.1.10,
  and taskrail v0.4.0. Repeated setup left exactly one profile marker.
- Ruff and mypy passed; all 46 tests passed. CLI help ran successfully.
- Independent General review: "No concrete task-relevant findings."

## Implementation Notes

- Reuse mise run setup; keep model downloads and Ollama optional at runtime.
- Simplification review made no changes; candidate validation had no findings.
- 2026-09-20T14:24:14Z: verification pass
- 2026-09-20T14:24:14Z: Orb lifecycle implementation verified locally; activation awaits delivery to the default branch.
