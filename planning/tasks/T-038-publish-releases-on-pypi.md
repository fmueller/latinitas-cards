---
id: T-038-publish-releases-on-pypi
title: Publish releases on PyPI
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies:
    - T-061-verify-managed-csv-import-safety-and-recovery-in
    - T-062-verify-morphology-presentation-on-ankimobile-and
    - T-060-generate-precision-gated-latin-form-parsing
    - T-068-publish-v0-2-0
updated_at: "2026-10-03T10:12:32Z"
---

# T-038-publish-releases-on-pypi Publish releases on PyPI

## Description

Deliver PyPI publishing for v0.3.0 and subsequent approved releases, following
`specs/v0.3.0.md#pypi-publishing` (moved from v0.2.0 on 2026-10-06). Add the release
workflow and installation guidance; actual publication requires explicit maintainer
approval.

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
- Before the first PyPI publication, require completed native update and presentation
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

- Publication depends on T-061's native update evidence, T-062's native
  presentation evidence, and T-060's form-parsing work and affected native retests.
  T-051's claim-policy and T-048's destination-contract are transitive
  prerequisites, but their release-candidate evidence remains an explicit
  publication requirement above. Workflow/build preparation may proceed earlier;
  this task cannot complete publication before those gates pass.
