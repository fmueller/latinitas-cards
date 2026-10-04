---
id: T-044-escape-terminal-controls-in-authored-export-paths
title: Escape terminal controls in authored export paths
status: todo
priority: high
spec_ref: specs/v0.1.1.md#authored-note-selection-and-preview
dependencies: []
updated_at: "2026-10-04T15:38:03Z"
---

# T-044-escape-terminal-controls-in-authored-export-paths Escape terminal controls in authored export paths

## Description

Fix the medium-severity terminal-output boundary missed by T-042 and found
during the second synthetic whole-spec adversarial test. Successful authored
export prints `Wrote {path}` verbatim. A supplied output directory containing
an OSC title payload therefore sends raw ESC/BEL bytes to the terminal, even
though the export succeeds. Escape the displayed path with the existing
terminal encoder without changing the actual destination or CSV content.

Evidence: https://ampcode.com/threads/T-01a10766-4ccf-73cd-816a-6bc2b46bd93d

## Acceptance

- Strict red/green regression uses a valid synthetic authored item and an
  existing output directory containing `out\u001b]0;OWNED\u0007`; current
  successful export emits raw ESC/BEL before the fix.
- Successful export status messages encode unsafe controls in displayed paths
  using `encode_unsafe_controls(..., preserve_line_breaks=False)`. No raw
  ESC/BEL or control-generated extra status lines reach terminal output.
- Actual export still succeeds at the exact user-supplied path, with the same
  deterministic CSV bytes, field mapping, provenance and stable identities;
  do not reject or sanitize the filesystem path itself.
- Tests cover safe ordinary paths and control-bearing paths, including line
  breaks, and preserve existing error/diagnostic encoding from T-042.
- Reproduce the original case with the real CLI, capture output safely, and
  verify both escaped terminal bytes and successful CSV creation.
- Keep scope to this finding; invalid-kind behavior remains an advisory and
  is not changed by this task.
- Complete workflow-v3 review and disposition verification; `uv run ruff check`,
  `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Record decisive regression results and the Taskrail verification timestamp;
  do not commit references to gitignored artifact paths.

## Implementation Notes
