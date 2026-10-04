---
id: T-042-escape-terminal-controls-in-authored-diagnostics
title: Escape terminal controls in authored diagnostics
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-selection-and-preview
dependencies: []
updated_at: "2026-10-04T13:29:28Z"
---

# T-042-escape-terminal-controls-in-authored-diagnostics Escape terminal controls in authored diagnostics

## Description

Fix the medium-severity terminal-control injection found during synthetic,
adversarial v0.1.1 testing. An unknown JSON property containing an ESC/BEL
sequence is included verbatim in a line-level issue and echoed by the authored
CLI. Protect the terminal display boundary using the existing control encoder
without changing structured diagnostics or source values.

Evidence: https://ampcode.com/threads/T-01a106d3-a5dc-7716-b644-a1b50079ec6d

## Acceptance

- Reproduce an unknown property named `bad\u001b]0;OWNED\u0007` followed by
  a valid JSONL row; prove the current CLI emits raw terminal controls before
  fixing it through strict red/green TDD.
- Authored validate, preview, and export diagnostics escape unsafe terminal
  controls, including attacker-controlled field paths and error text, using
  the existing encoder at the terminal boundary. No raw ESC/BEL reaches output.
- Errors still identify the source line and failed assumption, retain later
  valid rows diagnostically, and exit nonzero. Structured provenance and
  opaque source values remain unchanged.
- Invalid exports create no new output and leave existing output byte-identical.
  Existing safe card display and ordinary multiline formatting remain correct.
- Exercise the actual CLI with synthetic adversarial data, not only helper tests.
- Complete workflow-v3 review and disposition verification; `uv run ruff check`,
  `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Record decisive regression results and the Taskrail verification timestamp;
  do not commit references to gitignored artifact paths.

## Implementation Notes

- 2026-10-04T13:29:28Z: verification pass
- Terminal-boundary encoding covers issue fields/assumptions and caught input/export
  errors; structured issues, source values, and report formatting remain unchanged.
- Strict red: seven regression cases failed on raw controls before the production
  change. Green: 19 focused authored-preview tests passed after simplification.
- Dedicated simplifier removed two meaningless skipped test combinations only.
  Independent General, Security, and Python lanes each concluded:
  "No concrete task-relevant findings." Fresh candidate validation and disposition
  verification confirmed no findings, rejections, deduplication, or deferrals.
- Real subprocess validate/preview/export checks each exited 1, emitted no raw
  ESC/BEL, identified line 1, recovered one later valid diagnostic row, created no
  files, and left input and existing CSV byte-identical.
- Final exact chain: ruff check passed; mypy passed for 65 source files;
  pytest -v passed all 515 tests. Verification timestamp: 2026-10-04T13:29:28Z.
