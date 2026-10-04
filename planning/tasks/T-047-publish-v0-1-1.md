---
id: T-047-publish-v0-1-1
title: Publish v0.1.1
status: in_progress
priority: medium
spec_ref: specs/v0.1.1.md#goals
dependencies: []
updated_at: "2026-10-04T21:23:19Z"
---

# T-047-publish-v0-1-1 Publish v0.1.1

## Description

Publish the completed authored-note workflow from specs/v0.1.1.md using the
existing annotated-tag and GitHub-release process. Owner authorization covers
release metadata commits, main pushes, the tag, and GitHub publication only.

## Acceptance

- Synchronize package/lock version, dated user-focused changelog, and installation docs.
- Pass mandatory ruff/mypy/pytest, local policy gates, and exact-commit supported-Python CI.
- Review existing native Anki evidence and assess current dependency advisories.
- Publish one annotated v0.1.1 tag and matching GitHub release; do not publish to PyPI.
- Verify fresh-tag locked installation and record publication evidence separately.

## Verification Notes

- Metadata regression failed as expected before the version update (0.1.0 != 0.1.1).
- Implementation readiness: cycles 4/5 PASS, 627 tests; native Anki 26.9.3 backend
  first/reimport checks retain stable note/card identities and nonempty Personal Notes.
- Open urllib3 alerts #141/#142/#143 affect optional annotation extras only; default
  locked export excludes urllib3/requests. Exposure and limits documented for release.
- Metadata regression then passed; final exact ruff/mypy/pytest chain passed
  (65 source files, 627 tests). Full mise check, Taskrail validation, and diff check pass.
- Dedicated simplifier removed unrelated lock marker churn; only project version
  changed. General independent review, candidate validation, and fresh disposition
  verification each reported: "No concrete task-relevant findings." No deferrals.
- General lane only: no runtime, security implementation, persistence, or language
  changes. Publication remains gated by exact-commit remote CI and tag verification.

## Implementation Notes

- This task does not activate v0.2.0 or implement T-038 PyPI publishing.
- 2026-10-04T21:23:19Z: verification pass
