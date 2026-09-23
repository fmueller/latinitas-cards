---
id: T-017-optional-cltk-extra
title: Make CLTK annotation an optional extra
status: completed
priority: high
spec_ref: specs/v0.1.0.md#existing-experimental-capabilities
dependencies: []
updated_at: "2026-09-23T20:21:55Z"
---

# T-017-optional-cltk-extra Make CLTK annotation an optional extra

## Description

Move CLTK from the required dependencies into an optional `annotate` extra, as allowed by
`specs/v0.1.0.md#existing-experimental-capabilities`: annotation is experimental and not a
v0.1.0 release promise, yet CLTK pulls Stanza, PyTorch, CUDA wheels, and NLTK (with open
security advisories) into every install.

## Acceptance

- `pyproject.toml` declares `cltk` only under `[project.optional-dependencies] annotate`; a
  default `uv sync` installs neither CLTK, Stanza, nor PyTorch.
- Every command other than `annotate` works, and the CLI module imports, without CLTK.
- `annotate` without CLTK fails with an actionable error naming the `annotate` extra; when
  CLTK is installed but fails to import (for example a missing CUDA library), the error
  reports the underlying import failure instead of claiming CLTK is missing.
- Locked package versions are unchanged apart from extra markers.
- README, AGENTS.md, and the spec describe the extra and how to install it.
- ruff, mypy, and pytest pass without the extra installed.

## Verification Notes

- Verify run 2026-09-23T20:21:55Z passed: ruff, format, mypy, and pytest (61 tests) pass
  without the extra on Python 3.12, pytest passes on 3.10, and with the extra mypy and pytest
  pass and `annotate` produces analyses. A default sync removes CLTK, Stanza, PyTorch, and
  NLTK; locked package versions are unchanged.

## Implementation Notes

- 2026-09-23T20:21:55Z: verification pass
