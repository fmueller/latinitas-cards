---
id: T-023-preview-and-select-authored-notes-before-export
title: Preview and select authored notes before export
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-selection-and-preview
dependencies:
    - T-022-render-authored-note-kinds-into-dedicated-note
updated_at: "2026-10-04T10:45:07Z"
---

# T-023-preview-and-select-authored-notes-before-export Preview and select authored notes before export

## Description

Add side-effect-free validation and preview for an import file, plus selection filters
by kind, source section, source reference, and tag.

## Acceptance

- Preview reports counts by kind, source section, source reference, and status; merged
  duplicates; invalid or conflicting items with reasons; and representative cards.
- Validation and duplicate reconciliation cover the whole file before filtering. Preview
  reports recoverable errors together and exits nonzero on any error; any rendered cards
  then are diagnostic, not an exportable selection.
- Filters narrow the selection without modifying the import file; the effective selection
  is reported. Filters operate on merged items, including combined tags and filled references.
- `skip` items are counted but never selected for export.
- Preview consumes an internal typed result separate from terminal rendering.
- Tests prove preview writes no files and leaves the import file unchanged.
- CLI tests cover combined filters, filtering after duplicate merging, empty selection,
  and invalid skipped or filtered-out rows. Document validation and preview invocations
  with an explicit collection namespace and no corpus resources or principal-part mappings.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Verification pass at 2026-10-04T10:44:25Z, after simplification, independent
  General/Python/Security review, candidate validation, and fresh disposition review.
- Initial red: missing authored_preview module prevented test collection; green:
  six focused tests passed. Final focused suite: 12 passed.
- Final gate: `uv run ruff check`, `uv run mypy` (62 source files), and
  `uv run pytest -v` (500 passed). `git diff --check` clean.
- Manual CLI validation, combined filters on merged references/tags, empty skipped
  selection, and aggregate invalid skipped/filtered row diagnostics matched the
  expected exit codes (0/1); input SHA256 hashes stayed unchanged.
- Asymmetric decoys for each combined filter failed all four tests under an
  intentional AND-to-OR regression, then passed after restoring the correct logic.
- General and Python: "No concrete task-relevant findings." Security S-1:
  "Sanitize terminal control sequences in rendered card text before echoing it;
  HTML escaping does not prevent terminal escape-sequence interpretation."
  Candidate validation confirmed OSC bytes through a PTY. Fixed at the CLI
  boundary with the existing control encoder; the regression failed on raw ESC
  before the fix and passed afterward. Fresh review: "S-1 — RESOLVED"; no new
  task-relevant findings. No deferred findings.

## Implementation Notes

- 2026-10-04T10:44:25Z: verification pass
- `authored validate` and `authored preview` require an explicit namespace and
  use no corpus/profile mappings. Frozen typed preview results separate selection
  and diagnostics from terminal formatting. Counts describe reconciled,
  nonconflicting notes; all invalid rows/conflicting groups remain diagnostics.
- Filter values are exact; OR within dimensions, AND across dimensions, applied
  after whole-file reconciliation. Skip dominates merged status and never selects.
  Any error withholds exportable notes while permitting diagnostic card examples.
- One fixed-template representative front/back is shown per matching kind.
  Terminal controls are visible escaped text; internal rendered content is preserved.
  No output writer or CSV export was added (T-024 remains separate).
- Dedicated code-simplifier found no safe simplification and made no edits.
  Review lanes: General for acceptance/domain, Python for language/contracts,
  Security for untrusted input/terminal output. Database, framework, and ML lanes
  omitted because those boundaries were unchanged.
- 2026-10-04T10:45:07Z: Implemented and independently reviewed; verification pass at 2026-10-04T10:44:25Z; 500 tests and manual CLI checks pass
