---
id: T-022-render-authored-note-kinds-into-dedicated-note
title: Render authored note kinds into dedicated note types
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-kinds
dependencies:
    - T-020-parse-and-validate-authored-note-import-files
    - T-021-derive-stable-identities-for-authored-notes
updated_at: "2026-10-04T10:22:20Z"
---

# T-022-render-authored-note-kinds-into-dedicated-note Render authored note kinds into dedicated note types

## Description

Map `vocab`, `form`, and `qa` items onto one dedicated, stable note type each, sharing
the v0.1.0 note contract, with the default card directions from the spec.

## Acceptance

- Each kind has a stable note type name and field order with `LatinitasID` first,
  managed content fields, provenance, generation metadata, and a personal-notes field.
- `vocab` renders Latin to meaning; `form` renders the form (with context when present) to
  base form, analysis, and translation; `qa` renders question to answer.
- Contextually distinct occurrences of the same text form use distinct explicit keys.
  Tests show these yield separate items even with the same form text, while a citation
  edit alone preserves an established key's identity.
- Reference field definitions and front/back templates document creation of each stable
  note type with one card per authored item, including the user-owned Personal Notes field.
- Rendering is covered by tests for each kind, including optional fields left empty.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Verification passed at 2026-10-04T10:22:20Z, after all workflow-v3 gates.
- RED: `uv run pytest tests/unit/authored_notes_test.py -q` failed collection
  with `ModuleNotFoundError: No module named 'latinitas_cards.authored_notes'`.
  GREEN: the initial implementation passed all five contract cases. The sixth
  HTML-safety case failed under deliberate raw-text regression (1 failed,
  5 deselected); restoring escaping/newline handling passed all six cases.
- Exact final chain: `uv run ruff check` — All checks passed!;
  `uv run mypy` — Success: no issues found in 59 source files;
  `uv run pytest -v` — 488 passed in 8.82s. The fresh disposition reviewer
  independently reran the chain: 488 passed in 8.06s.
- Chromium flat-template harness: six populated/empty front/back pairs;
  DOM checks reported DPR 2, six pairs, two context blocks (front and back
  of the populated form), zero script elements and zero empty optional blocks.
  Screenshot inspection found correct directions, readable literal HTML text,
  no unresolved template tokens and no clipping. This is not native Anki
  import verification. Reviewable screenshot retained in the Amp thread.

## Implementation Notes

- Added a focused authored-note contract module, tests and generated reference
  documentation. Existing v0.1.0 schemas, templates and CLI remain unchanged.
  Authored schema `authored-1` reuses shared field ownership and generator/profile
  metadata; one Recognition template per kind, with Personal Notes never exported.
- Dedicated code-simplifier Task loaded the skill and simplified managed-field
  construction only. Accepted after inspecting the edit and rerunning six tests.
- Separate read-only General, Python and Security Tasks loaded code-reviewer and
  their mapped guidance (Python patterns, Security review). General and Security
  each concluded: "No concrete task-relevant findings." Security was selected
  for the plain-text-to-HTML boundary and Personal Notes safety; Python for the
  new typed module; General for acceptance/architecture. Database/framework lanes
  omitted: no database, migration, web framework or persistence implementation.
- Python candidate, verbatim:
  "render_authored_note can render a diagnostic item from a reconciliation result
  that failed validation, despite its contract requiring a validated reconciled
  item."
  Evidence: `src/latinitas_cards/authored_notes.py:104-114` accepts
  `IdentifiedAuthoredItem` directly; diagnostic notes expose the same type.
  Proposed direction was a validated-result or item wrapper.
- Fresh candidate-validation Task: "REJECTED — Finding 1". The existing renderer
  precondition requires upstream `require_valid()`, diagnostic items are explicitly
  non-exportable until that gate, and T-022 adds no bypassing selection/export
  entry point. The spec also permits diagnostic preview rendering with errors.
  No validated findings, no fixes or deferrals. Fresh disposition-verification
  Task independently confirmed rejection and concluded: "No concrete task-relevant
  findings." One review cycle; no outstanding review findings.
- 2026-10-04T10:22:20Z: verification pass
- 2026-10-04T10:22:20Z: Completed after strict TDD, dedicated simplification, independent General/Python/Security review, candidate validation, fresh disposition verification, and exact ruff/mypy/pytest gate (488 passed).
