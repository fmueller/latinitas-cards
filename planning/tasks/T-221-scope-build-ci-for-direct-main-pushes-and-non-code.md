---
id: T-221-scope-build-ci-for-direct-main-pushes-and-non-code
title: Scope build CI for direct main pushes and non-code changes
status: completed
priority: low
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies: []
updated_at: "2026-11-07T20:25:18Z"
---

# T-221-scope-build-ci-for-direct-main-pushes-and-non-code Scope build CI for direct main pushes and non-code changes

## Description

Scope Build CI for direct pushes to main and pull requests. Avoid the Python
matrix for documentation-only and Taskrail-only changes without weakening
application, packaging or infrastructure validation. Preserve local gates.

## Acceptance

- Compare the entire pushed range on main and the merge-base range for PRs.
- Documentation-only changes skip application checks; planning/spec/config
  changes run Taskrail validation. Unknown paths and unavailable comparisons
  run the full build; manual dispatch always runs the full build.
- Application, tests, dependency, packaging, hook and workflow changes retain
  lint/type checks, infrastructure guards and both Python test/package lanes.
- An always-reporting aggregate check fails when selected checks fail or skip
  unexpectedly. Local hooks and mandatory agent validation remain unchanged.
- Regression cases cover mixed changes, multiple commits, renames, deletions,
  unusual filenames and unavailable base revisions.

## Verification Notes

- Final mandatory chain passed in order: `uv run ruff check`, `uv run mypy`
  (199 source files), `uv run pytest -v` (3262 passed, 7 skipped, one warning).
- `bash scripts/check-ci-scope-test.sh`, shell syntax, YAML parsing, Taskrail
  validation, changed-Python formatting and `git diff --check` passed.
- All seven existing policy/infrastructure guard suites passed, including
  distribution inspection and publication boundaries.
- Initial full run exposed the packaging contract test's old single-dependency
  assertion. Updated it to require both scope selection and lint success while
  preserving both Python versions, install checks and no-publication assertions;
  restarted the entire mandatory chain successfully.
- Simplification and independent scheduling/tool-install and shell/security
  reviews found no implementation defects. The review coordinator reconciled
  specialist evidence and approved with no unresolved required findings.
- Review found a PR test gap: tip-only comparison survived the original fixture.
  Added an application commit followed by a docs commit on a divergent PR.
  The normal suite passes; an in-memory `PR_HEAD^` mutant now fails on the
  expected full-vs-docs scope assertion. No deliberate regression was retained.
- GitHub-hosted scheduling and cold-runner installation were not executed.
  Changes are local on main; no branch protection or remote state was changed.

## Implementation Notes

- Package metadata inputs such as README.md and licenses stay on the full lane.
- Documentation allowlist: `docs/*.md` (including nested Markdown), AGENTS.md,
  CLAUDE.md and CHANGELOG.md. Other paths default to full validation unless
  under planning/, specs/ or .taskrail/, which require Taskrail validation.
- Classification is path-based. Docs-path symlinks are not separately classified
  by mode; review found no concrete application-execution bypass from that alone.
- Local pre-commit/pre-push hooks and mandatory agent checks are unchanged.
- 2026-11-07T20:25:18Z: verification pass
- 2026-11-07T20:25:18Z: CI scope implementation and local validation complete; not committed or pushed.
