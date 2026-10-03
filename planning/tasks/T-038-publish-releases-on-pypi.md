---
id: T-038-publish-releases-on-pypi
title: Publish releases on PyPI
status: todo
priority: medium
spec_ref: specs/v0.2.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-03T10:12:32Z"
---

# T-038-publish-releases-on-pypi Publish releases on PyPI

## Description

Deliver PyPI publishing for v0.2.0 and subsequent approved releases, following
`specs/v0.2.0.md#pypi-publishing`. Add the release workflow and installation guidance;
actual publication requires explicit maintainer approval.

## Acceptance

- Build and validate source distributions and wheels from the release commit.
- Verify clean installation of both distributions and the `latinitas-cards --help`
  entry point without optional annotation extras or CLTK/Stanza resources.
- Use PyPI Trusted Publishing with approved release tags and a protected publishing
  environment; routine CI cannot publish, and publication follows passing release
  checks and explicit maintainer approval.
- Confirm PyPI package availability and Trusted Publisher configuration before release.
- Include complete package metadata and bundled licensing notices, and document
  installation, upgrades, supported Python versions, and optional extras.
- After approved publication, verify installation of the exact release from PyPI.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
