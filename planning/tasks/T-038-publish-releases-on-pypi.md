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
- Inspect both artifacts' contents and metadata for required licensing notices,
  runtime presentation resources, and release version/tag consistency. Build a
  wheel from the extracted source distribution as well.
- Verify clean installation of both distributions on supported Python 3.13 and
  3.14 outside the checkout. Run `latinitas-cards --help` and a sanitized core
  workflow; confirm default installation does not install CLTK/Stanza or optional
  annotation extras and does not require their runtime resources.
- Use PyPI Trusted Publishing with approved release tags and a protected publishing
  environment; routine CI cannot publish, and publication follows passing release
  checks and explicit maintainer approval.
- Before v0.2.0 publication, require completed native update and presentation
  evidence gates and form-parsing work, including its affected native retests.
  Require evaluated claim-policy evidence and the documented supported-capability
  matrix. Bind the evidence to the release candidate; repeat affected checks when
  relevant changes invalidate it. Unavailable evidence remains an open gate and
  maintainer approval cannot substitute for it. Workflow/build preparation may
  proceed independently of these publication gates.
- Confirm PyPI package availability and Trusted Publisher configuration before release.
- Include complete package metadata and bundled licensing notices, and document
  installation, upgrades, supported Python versions, and optional extras.
- Document optional-extra behavior for pip without implying that uv's CPU index
  selection transfers to pip. Publish the same validated artifacts after explicit
  maintainer approval.
- After approved publication, verify installation of the exact release from PyPI.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- The decomposition remains draft-only in `planning/v0.2.0-task-draft.json`.
  When imported task IDs exist, add dependencies on the tasks with draft keys
  `native-update-evidence`, `native-presentation-evidence`, and
  `form-parsing-exercises`; do not place unresolved draft keys in task frontmatter.
  Claim-policy and destination-contract are transitive prerequisites, but their
  release-candidate evidence remains an explicit publication requirement above.
