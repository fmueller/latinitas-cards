---
id: T-014-publish-v0-1-0
title: Publish v0.1.0
status: completed
priority: medium
spec_ref: specs/v0.1.0.md#release-readiness
dependencies:
    - T-009-prepare-v0-1-0-release-candidate
    - T-026-render-source-html-safely
    - T-027-protect-personal-notes-on-repeat-import
    - T-028-inherit-parent-anki-note-tags
    - T-035-verify-the-revised-note-architecture-before
updated_at: "2026-10-02T17:04:56Z"
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

- Owner approved publication on 2026-10-02, including the requested editorial
  changelog/guidance preparation; PyPI remains deferred.
- Published at 2026-10-02T17:00:06Z:
  [v0.1.0 release](https://github.com/fmueller/latinitas-cards/releases/tag/v0.1.0).
- Annotated [v0.1.0 tag](https://github.com/fmueller/latinitas-cards/tree/v0.1.0)
  points to the exact approved release-preparation
  [commit](https://github.com/fmueller/latinitas-cards/commit/183963bdb4e31bbe94a4763bbd84c0369c0698a5).
  Runtime, package metadata, and lock inputs are unchanged from the T-035 candidate;
  upstream T-037 planning changes were preserved. No released commit was rewritten.
- [Exact-commit CI](https://github.com/fmueller/latinitas-cards/actions/runs/37037576863)
  passed lint/type checking, guards, and Python 3.13/3.14 tests. Locked local
  ruff/mypy/pytest chains passed on both versions: 417 passed, 10 skipped each.
  The integrated `mise run check`, Taskrail validation, and `git diff --check` passed.
- A fresh clone of `v0.1.0` passed the README installation commands:
  `uv sync --locked` and `uv run latinitas-cards --help`.
- Release metadata assertions were checked red/green. Simplification made no edits;
  General review found no issues. The later release-status consistency finding was
  resolved by actual publication and rejected as obsolete in candidate validation.
  Existing T-035 synthetic Anki 26.09.3 evidence remains the native-client boundary;
  this editorial release preparation does not claim a new native Anki run.

## Implementation Notes

- Do not publish until the readiness dependency is completed and independently verified.
- 2026-09-27T08:56:37Z: Publication is gated by T-035-verify-the-revised-note-architecture-before and its T-029 through T-034 dependencies. Unblock only after the revised architecture gate passes and the owner approves the exact candidate; old single-card readiness evidence is not approval for this model.
- 2026-10-02T16:42:42Z: Owner explicitly approved v0.1.0 publication on 2026-10-02 after the completed T-035 gate, requesting a user-focused changelog and AGENTS.md guidance before release; PyPI remains deferred. Only editorial release preparation may change the verified runtime candidate.
- 2026-10-02T16:55:43Z: verification pass
- 2026-10-02T16:58:04Z: verification pass
- 2026-10-02T16:58:27Z: Rebase restored the old blocker projection; reapply the owner-approved publication transition through CLI before publishing.
- 2026-10-02T16:58:27Z: Owner approval on 2026-10-02 and completed T-035 still apply after integrating documentation-only T-037; clear the stale merged blocker.
- 2026-10-02T17:04:56Z: verification pass
- 2026-10-02T17:04:56Z: Published approved v0.1.0 at https://github.com/fmueller/latinitas-cards/releases/tag/v0.1.0; exact-commit CI and fresh-tag install pass. Publication evidence recorded separately without rewriting released commit. PyPI deferred.
