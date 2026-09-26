---
id: T-014-publish-v0-1-0
title: Publish v0.1.0
status: todo
priority: medium
spec_ref: specs/v0.1.0.md#release-readiness
dependencies:
    - T-009-prepare-v0-1-0-release-candidate
    - T-026-render-source-html-safely
    - T-027-protect-personal-notes-on-repeat-import
    - T-028-inherit-parent-anki-note-tags
updated_at: "2026-09-22T16:09:47Z"
---

# T-014-publish-v0-1-0 Publish v0.1.0

## Description

Publish the verified v0.1.0 release candidate without changing its contents between
approval, tagging, and GitHub release creation.

## Acceptance

- Reconfirm that the candidate commit is the exact commit approved by the readiness task
  and that required CI checks passed on supported Python versions.
- After T-026, T-027, and T-028, revalidate the revised candidate with the full
  readiness checks and native Anki first/repeat import and rendering checks.
  Obtain user approval of that exact candidate; prior candidate checks and
  approval do not cover these changes automatically.
- `pyproject.toml`, changelog, release notes, and the annotated `v0.1.0` tag use the same
  version.
- The tag points to the approved candidate commit.
- A GitHub release exists for `v0.1.0` with release notes that describe the stable
  deck-first scope and explicit experimental exclusions.
- Publication evidence records the tag and release URLs without modifying the released
  commit.

## Verification Notes

- TODO: record verification evidence and publication timestamp.

## Implementation Notes

- Do not publish until the readiness dependency is completed and independently verified.
