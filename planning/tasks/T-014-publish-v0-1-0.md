---
id: T-014-publish-v0-1-0
title: Publish v0.1.0
status: blocked
priority: medium
spec_ref: specs/v0.1.0.md#release-readiness
dependencies:
    - T-009-prepare-v0-1-0-release-candidate
    - T-026-render-source-html-safely
    - T-027-protect-personal-notes-on-repeat-import
    - T-028-inherit-parent-anki-note-tags
    - T-035-verify-the-revised-note-architecture-before
updated_at: "2026-09-27T08:56:37Z"
---

# T-014-publish-v0-1-0 Publish v0.1.0

## Description

Publish the verified v0.1.0 release candidate without changing its contents between
approval, tagging, and GitHub release creation.

## Acceptance

- Reconfirm that the candidate commit is the exact commit approved by the readiness task
  and that required CI checks passed on supported Python versions.
- Require T-035 and all its dependencies, including T-036 interruption recovery and the
  identity/normalized-eligibility/schema regressions in T-029/T-032/T-033. The original
  blocker remains in force; its earlier dependency list is extended by this graph.
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
- 2026-09-27T08:56:37Z: Publication is gated by T-035-verify-the-revised-note-architecture-before and its T-029 through T-034 dependencies. Unblock only after the revised architecture gate passes and the owner approves the exact candidate; old single-card readiness evidence is not approval for this model.
