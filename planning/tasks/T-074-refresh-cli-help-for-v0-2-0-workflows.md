---
id: T-074-refresh-cli-help-for-v0-2-0-workflows
title: Refresh CLI help for v0.2.0 workflows
status: completed
priority: medium
spec_ref: specs/v0.2.0.md#goals
dependencies: []
updated_at: "2026-10-07T23:14:07Z"
---

# T-074-refresh-cli-help-for-v0-2-0-workflows Refresh CLI help for v0.2.0 workflows

## Description

Found by the T-069 review. Top-level `latinitas-cards --help` still describes only
split/annotate/cloze; `preview`/`generate` one-liners describe only USFX clozes although
`--profile` is the primary principal-part path; `form-parsing export --approve-fresh-import`
has no help; the generate checkpoint error does not name `--approve-fresh-import`. Not release-blocking.

## Acceptance

- Top-level, preview and generate help name the profile/principal-part workflow and mark USFX cloze options experimental.
- `form-parsing export --approve-fresh-import` and the generate checkpoint error name the flag and its effect.
- CLI registration/help tests updated; mandatory chain passes.

## Verification Notes

- Verification pass recorded at 2026-10-07T23:14:07Z after fresh `uv run ruff check`,
  `uv run mypy` (87 source files), and `uv run pytest -v` (855 passed).
- Strict CLI-output TDD: five failures before implementation, then five passes.
  Initial full-chain wording failure was fixed and the full chain restarted.
- Simplifier: no edits; 75 focused tests passed. Separate General and Python lanes
  both reported "No concrete task-relevant findings." Candidate validation and
  fresh disposition verification found none; no fixes, deferrals, or follow-ups.
- Installed root, preview, generate, and form-parsing export help inspected.
  Isolated package smoke: checkpoint gate exited 2 with full flag/effects and no
  output; explicit fresh-import approval exited 0 and produced CSV.

## Implementation Notes

- 2026-10-07T23:14:07Z: verification pass
- 2026-10-07T23:14:07Z: Help-only workflow refresh; CLI TDD, simplifier, independent General/Python lanes, candidate/disposition verification, installed CLI smoke and fresh ruff/mypy/855-test chain passed.
