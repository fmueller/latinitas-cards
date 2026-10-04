---
id: T-024-export-selected-authored-notes-as-deterministic
title: Export selected authored notes as deterministic CSV
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-selection-and-preview
dependencies:
    - T-023-preview-and-select-authored-notes-before-export
    - T-004-preview-and-deterministic-csv-export
updated_at: "2026-10-04T11:12:37Z"
---

# T-024-export-selected-authored-notes-as-deterministic Export selected authored notes as deterministic CSV

## Description

Write the selected authored notes through the v0.1.0 CSV export boundary so repeated
imports update the same logical notes.

## Acceptance

- Export writes UTF-8 Anki-import CSV per note type with stable field order, target deck,
  provenance tags, and import metadata or precise instructions.
- Same import file, namespace, and filters produce byte-stable output.
- Tests prove edited content keeps identity, `skip` items are absent, and Personal Notes
  is omitted from CSV columns rather than written empty; check headers and row widths for
  every kind. Import instructions leave Personal Notes unmapped.
- Invalid input anywhere in the file, including skipped or filtered-out rows and conflicting
  duplicates, exits nonzero before any output writes. CLI tests verify existing output
  stays unchanged and no new output is created. Empty selection reports zero without writes.
- Documentation covers first import, re-import after edits, and combined selection filters,
  with explicit namespace configuration and no corpus resources or principal-part mappings.
- Record a supported-Anki first-import/re-import check for each kind: edited content updates
  the same note, nonempty Personal Notes survives, and the note yields one intended card.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Strict TDD: `uv run pytest tests/unit/authored_export_test.py -q` first
  failed all six cases with `No such command 'export'`, then passed all six.
  Combined authored and existing export coverage passed 87 tests.
- Initial full chain: Ruff passed; mypy passed for 64 source files; pytest
  passed 506 tests. An outdated callback-registration expectation was updated
  to include export after the initial full suite reported 1 failed, 505 passed.
- Dedicated code-simplifier Task loaded its skill and made no changes; six
  authored-export tests passed again. General, Security, Python, and Database
  reviewers ran as separate parallel read-only Tasks with code-reviewer and
  their routed ECC/companion guidance. Each returned verbatim:
  `No concrete task-relevant findings.`
- Fresh candidate-validation Task loaded code-reviewer: zero accepted,
  rejected, or deduplicated candidates; `No concrete task-relevant findings.`
  No finding dispositions or deferrals are needed.
- Fresh disposition-verification Task loaded code-reviewer and confirmed:
  `No concrete task-relevant findings.` It independently reran the exact
  Ruff/mypy/pytest chain: 506 tests passed. One review cycle, zero unresolved
  findings, no fixes or deferrals. Finalization follows the final full chain.
- Real subprocess CLI checks: combined kind/section/reference/tag selection
  reported two and wrote only vocab/form; repeating it produced identical
  SHA-256 hashes. Invalid skipped QA, filtered form, and conflicting vocab
  rows reported all three errors, exited 1, and retained prior file hashes.
  Empty selection reported zero and did not create the absent directory.
- Actual Anki 26.09.3 native backend on Linux (PyPI anki==26.9.3):
  `uv run --with anki==26.9.3 python scripts/check-authored-anki.py` passed.
  Each kind first imported one note/Recognition card; fresh-metadata edited
  re-import retained note ID, LatinitasID, card ID, deck, and nonempty Personal
  Notes. Meaning/Translation/Answer changed as expected, HTML-safe rendering
  passed, total remained three notes/cards. Database reviewer independently
  reran this native gate. This is not a desktop-dialog interaction claim.

## Review Scope

- General covers correctness, acceptance, and documentation; Security covers
  input trust, HTML/header injection, output paths, and data loss; Python
  covers typing, error semantics, and tests; Database covers persisted shared
  transaction/recovery and disposable native Anki persistence. Three specialist
  lanes fit the default budget. No framework, ML, RAG, healthcare, or network
  lane is triggered. No production database schema or migration changes.

## Implementation Notes

- Authored selection/rendering stays in its existing modules; a focused
  authored exporter supplies managed rows to the shared CSV serializer and
  recoverable file transaction, preserving principal-parts export guards.
- Export requires explicit namespace, deck, and an existing output directory.
  It omits empty kinds and Personal Notes, rejects input aliases and symlink
  destinations, and prepares every payload before committing any file.
- Reference templates and appearance are unchanged; native rendered question
  and answer checks validate content without claiming new UI visual work.
- 2026-10-04T11:12:27Z: verification pass
- 2026-10-04T11:12:37Z: Verified pass at 2026-10-04T11:12:27Z after final ruff/mypy/pytest chain (506 tests), actual native Anki 26.09.3 first/reimport checks for each kind, manual CLI checks, simplification, four independent reviewer lanes, candidate validation and fresh disposition verification; no unresolved findings.
