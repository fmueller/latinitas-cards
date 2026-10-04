---
id: T-044-escape-terminal-controls-in-authored-export-paths
title: Escape terminal controls in authored export paths
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-selection-and-preview
dependencies: []
updated_at: "2026-10-04T15:48:12Z"
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
- Strict RED: `uv run pytest -q tests/unit/authored_export_test.py -k
  status_escapes` produced 2 failed, 1 passed: original OSC ESC/BEL and
  line-control paths failed their exact displayed-status assertions.
- GREEN: authored export and preview suites passed all 36 tests after the
  display-only fix; all three note kinds retain byte-identical CSV output.
- Real subprocess CLI checks used `capture_output=True` for ordinary Unicode,
  original OSC, and LF/CR/tab/U+2028/U+2029 directories. All exited zero with
  empty stderr, nine output lines, exact escaped status tails, CSV creation at
  the supplied paths, identical baseline CSV bytes, and unchanged input bytes.
- Dedicated code-simplifier Task loaded its skill, made no edits, and passed
  the 36 focused tests. Separate General, Security, and Python reviewer Tasks
  loaded code-reviewer and their lane guidance. Each concluded verbatim:
  "No concrete task-relevant findings." Fresh candidate validation retained
  none and rejected none; fresh disposition verification confirmed no unresolved
  or newly introduced findings. One review cycle; no fixes or deferrals.
- Security is required by the terminal-output trust boundary; Python covers
  implementation/test semantics. No database, framework, or other domain lane
  is materially affected. Existing diagnostics, Unicode validation, and the
  invalid-kind advisory remain unchanged.
- Final exact chain passed: `uv run ruff check` (All checks passed!),
  `uv run mypy` (no issues in 65 source files), `uv run pytest -v`
  (560 passed in 9.15s). `git diff --check` also passed.

## Implementation Notes

- Encode only the displayed successful-export path with the existing encoder
  and `preserve_line_breaks=False`; filesystem paths and writer are unchanged.
- 2026-10-04T15:48:02Z: verification pass
- 2026-10-04T15:48:12Z: Verified pass at 2026-10-04T15:48:02Z after strict red/green, independent review and disposition verification, captured real CLI checks, and exact ruff/mypy/pytest chain (560 passed).
