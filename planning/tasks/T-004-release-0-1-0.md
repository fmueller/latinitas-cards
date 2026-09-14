---
id: T-004-release-0-1-0
title: Add a changelog and cut the 0.1.0 release
status: todo
priority: medium
spec_ref: specs/v0.1.0.md#release-hygiene
dependencies:
    - T-001-cloze-difficulty
    - T-002-parallel-translations
updated_at: "2026-09-14T09:14:51Z"
---

# T-004-release-0-1-0 Add a changelog and cut the 0.1.0 release

## Description

`pyproject.toml` declares `0.1.0`, but there is no `CHANGELOG.md`, no `v0.1.0` tag, and no
GitHub release — the version is written but unreleased. Cutting the release means documenting
the shipped command set and confirming the docs match the code.

## Acceptance

- `CHANGELOG.md` exists in Keep a Changelog format with a `0.1.0` entry covering the shipped command set.
- README and AGENTS.md match the shipped commands and options.
- A `v0.1.0` tag exists and matches the `pyproject.toml` version.
- The full chain (`ruff check`, `mypy`, `pytest -v`) is green on a clean checkout before tagging.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
