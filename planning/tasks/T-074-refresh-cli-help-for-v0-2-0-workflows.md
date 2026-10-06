---
id: T-074-refresh-cli-help-for-v0-2-0-workflows
title: Refresh CLI help for v0.2.0 workflows
status: todo
priority: medium
spec_ref: specs/v0.2.0.md#goals
dependencies: []
updated_at: "2026-10-06T20:16:22Z"
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

- Pending.

## Implementation Notes


