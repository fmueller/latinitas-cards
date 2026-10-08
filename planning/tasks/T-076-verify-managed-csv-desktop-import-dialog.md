---
id: T-076-verify-managed-csv-desktop-import-dialog
title: Verify managed CSV updates through the Anki Desktop import dialog
status: completed
priority: medium
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-08T00:21:05Z"
---

# T-076-verify-managed-csv-desktop-import-dialog Verify managed CSV updates through the Anki Desktop import dialog

## Description

Found by the T-069 review. T-061 verified managed CSV only through Anki's backend import API;
the documented user path is the Desktop import dialog, which earlier skipped tag-only rows.
Disclosed as a v0.2.0 limitation. Not release-blocking.

## Acceptance

- Native Desktop dialog runs on a sanitized fixture record client/version, settings, content, tag-only and no-op outcomes with full card/review-log comparisons.
- Skipped tag-only effects remain pending/unresolved, and docs state the verified scope.

## Verification Notes

- 2026-10-08: Actual Anki Desktop 26.09.3 (aqt/anki 26.9.3), Python 3.14.2,
  PyQt6 6.11.0/Qt 6.11.2, Linux x86_64/glibc 2.36 orb Desktop/Xwayland.
  `scripts/check-managed-desktop.py --prepare`, then `--gui`, `--drive`,
  `--finish` for content, tag-add, tag-remove, tag-final-remove, tag-unmapped,
  noop. Qt CSV ImportDialog controls and Import button use actual CDP pointer
  clicks; JavaScript only reads/scrolls DOM. No backend update proxy.
- Correct settings: comma/HTML forced by CSV, existing dedicated model/deck,
  Update / Note Type matching, first-field LatinitasID, named field columns,
  Personal Notes unmapped, Tags column 5, both extra tag inputs empty.
- Chosen Meaning and tag subset applied; unapproved lemma and second-note
  proposal stayed untouched. All three correctly mapped tag-only effects
  applied with identical fields: manual/source → added/manual/source,
  manual/source → manual, source → empty. This contradicts the older skip
  premise for this exact current fixture/client/settings, not all clients.
- All columns of all eight cards, fifteen revlog rows, note-type/field/template/
  deck/deck-config rows compare equal for every case. Every unapproved note
  column, personal text and manual tag survives. Only approved Meaning/tags and
  affected-note native mod/usn may differ; actual runs changed mod, not usn.
- Actual GUI-content result copied closed for no-op reapplication: native
  Skipped, zero full-table deltas, repeat observe leaves the journal identical.
  Deliberately Tags unmapped fault: native Skipped, zero deltas; actual observe
  classifies the emitted tag effect unresolved, observed/pending empty, anchors
  unchanged. Emission was pending before observation; unresolved is not success.
- Native Qt file/settings/mapping/result captures inspected for readable exact
  controls and results, including all tag-only outcomes, no-op and skipped fault.
  Complete runtime snapshots, closed backups, reports and evidence hashes bound
  by source/client bindings; no runtime artifact paths committed here.
- Executed harness SHA-256:
  `6524be57140faeb94d210b277829d57a78da962a8521ceef1e0f7572a150172b`.
  Base source is accepted remote commit 0170926; fetched containment and clean
  initial checkout confirmed. Application source hashes recorded per run.
- Existing `check-managed-capture.py` passed the installed CLI capture/adopt/
  plan/approve/emit/observe recipe, full card/history/model/deck tables unchanged.
- Initial full chain: `uv run ruff check`; `uv run mypy` (89 source files);
  `uv run pytest -v` (887 passed). Final full chain after task-local harness
  corrections: ruff passed; mypy 89 files passed; pytest 887 passed in 21.78s.

## Implementation Notes

- Workflow-v3 evidence-only adaptation: no production behavior changes or
  fictitious production RED/GREEN. Executable fixture assertions establish the
  required native outcomes. Real probe failures caught SQLite sidecars before
  capture, hidden/stale DOM dropdown selection, viewport-edge pointer delivery,
  Escape closing a Qt dialog and the mistaken pending/unresolved expectation;
  final clean six-case run uses closed-backup acquisition and actual pointers.
- Reuses existing native fixture/comparison helpers and current production
  capture/adoption/lifecycle; no reimplementation or domain policy change. New
  optional Desktop harness and focused verification doc, backend-doc/README
  crosslinks and corrected changelog client/scope limitation only.
- Dedicated code-simplifier skill: "No simplification warranted; no files
  changed." Rechecked ruff/mypy/pytest; 887 passed.
- Independent General, Database/persistence, Security and Python reviewers each
  loaded code-reviewer with dedicated guidance. Each returned verbatim:
  "No concrete task-relevant findings." Database covered SQLite lifetimes/full
  comparisons/journal invariants; Security covered fixture ownership/CDP and
  data-loss/truthful claims; Python covered runtime cleanup/assertions; General
  covered acceptance and docs. No framework lane: no application UI changed.
  Three specialist lanes fit the default budget; no product ML/network changes.
- Fresh candidate validation: "No concrete task-relevant findings." Accepted
  IDs: none; rejected IDs: none. No concrete findings to disposition or defer.
  Fresh disposition verification and final gates precede verify/complete.
- Self-audit correction (not a reviewer finding): RED hash audit failed with
  `AssertionError: ('content', 'finish.log')` because redirected stdout changed
  after hashing. Minimal fix hashes only stable JSON/CSV/PNG/closed database
  evidence, excluding logs/requests/markers; clean native rerun and independent
  source/evidence hash audit establish GREEN without changing product behavior.
- Native visual check also rejected an early no-op loading-spinner capture
  despite ready result DOM. Wait for two animation frames plus Qt presentation
  before the native capture, rerun the fixture and inspect the final result;
  screenshots are evidence of UI state, not substitutes for offline comparisons.
- Fresh disposition verifier: "PASS — no unresolved or newly introduced
  task-relevant findings". Independently matched all 49 source hashes, all six
  case manifests (17 stable files per effect case, 16 for no-op), archive hash
  and final native settings/mapping/no-op/skipped captures. No findings deferred.
  Verification was read-only; it did not claim to rerun native imports or gates.
- 2026-10-08T00:21:05Z: verification pass
- 2026-10-08T00:21:05Z: Native Desktop dialog evidence accepted after six-case run, full comparisons, source/evidence hash audit, inspected captures, independent reviews and final ruff/mypy/pytest gates. Exact client/settings scope only; no production policy change.
