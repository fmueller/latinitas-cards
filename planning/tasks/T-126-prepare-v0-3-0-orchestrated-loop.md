---
id: T-126-prepare-v0-3-0-orchestrated-loop
title: Prepare v0.3.0 contracts and backlog for an orchestrated implementation loop
status: completed
priority: high
spec_ref: specs/v0.3.0.md#goals
dependencies: []
updated_at: "2026-10-14T21:20:31Z"
---

# T-126-prepare-v0-3-0-orchestrated-loop Prepare v0.3.0 contracts and backlog for an orchestrated implementation loop

## Description

Prepare the active `specs/v0.3.0.md#goals` contracts and tracked backlog for a
medium-worker orchestrated implementation loop, as requested by the maintainer.
This is pre-loop planning work, not feature implementation or release authority.

## Acceptance

- Resolve reference grammar, apostrophe tokenization and corpus claim exit-code
  contradictions in the spec and dependent tasks.
- Define apply-time policy safety and recoverable two-file normalization with
  explicit regression scenarios in the implementing tasks.
- Align active-spec guidance, preserve the blocked maintainer release gate, and
  leave no preparation task active when handing off to T-102.
- Taskrail validation passes, all active-spec areas remain covered, and the
  mandatory ruff/mypy/pytest chain passes after integration.

## Verification Notes

- Integrated verification on 2026-10-14: `git diff --check` and Taskrail validation
  passed; all six active-spec areas remain covered with no orphans.
- The full mandatory chain passed: ruff clean, mypy clean (107 source files),
  pytest 1050 passed in 35.85 seconds. The first run exposed the old spec-index
  assertion; it now checks active-but-unreleased v0.3.0, and the full chain was rerun.
- Reviewed the three medium workers' transferred diffs. Updated 15 existing task
  files, preserving lifecycle statuses and adding only the necessary dependency edges.
- The T-103 dependency-anomaly signal is advisory: downstream tasks depend on this
  foundation through other spec areas; no missing prerequisite was found.

## Implementation Notes

- Use medium-mode sub-threads with disjoint spec, guidance and task-plan ownership.
- Transfer unpushed files explicitly and inspect their integrated diffs.
- Do not start implementation, unblock publication, push, tag or release.
- 2026-10-14T21:20:31Z: verification pass
- 2026-10-14T21:20:31Z: Pre-loop preparation complete; implementation remains unstarted and release gate stays blocked.
