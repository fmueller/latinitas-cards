---
id: T-075-capture-snapshots-and-first-adoption-from-cli
title: Capture destination snapshots and first adoption from the CLI
status: completed
priority: medium
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-07T23:38:55Z"
---

# T-075-capture-snapshots-and-first-adoption-from-cli Capture destination snapshots and first adoption from the CLI

## Description

Found by the T-069 review. `managed plan/emit/observe/reconcile` need snapshot, proposal and
baseline JSON, but no command captures a destination snapshot from a closed backup or performs
first adoption (`adopt()` is library-only); users must hand-assemble JSON. Disclosed as a
v0.2.0 limitation. Not release-blocking.

## Acceptance

- A documented, offline command builds a validated version-1 snapshot from a closed collection backup without opening a running collection.
- First adoption is available from the CLI with explicit per-note ownership review; visible text never adopts a note.
- README managed example works end to end on sanitized fixtures.

## Verification Notes

- Strict RED/GREEN: the initial 18 acquisition/adoption tests failed because the
  capture module/commands did not exist, then passed after implementation. Additional
  missing-column, stored-source, failed-journal-write and deck-config tests each
  reproduced unsafe acceptance before the corresponding narrow correction.
- Initial exact `uv run ruff check`, `uv run mypy`, `uv run pytest -v` gate passed
  (884 tests). The old exact command-registration assertion initially failed for
  the new capture/adopt commands and was updated; the full chain restarted.
- The documented installed sanitized recipe passed with optional `anki==26.9.3`:
  `work=$(mktemp -d); uv run --with anki==26.9.3 python scripts/check-managed-capture.py
  --output-dir "$work/evidence"`. It executes installed capture/adopt/plan/approve/
  emit/observe, reuses fixture-only native provisioning/import, compares every
  surviving card/history/model/deck column, personal text and exact manual tags,
  and verifies that emission alone does not advance the baseline.
- Dedicated `code-simplifier` Task loaded the personal skill, made no edits,
  rejected speculative row-index restructuring, and passed 80 focused tests,
  Ruff and mypy.
- Independent read-only Tasks loaded `code-reviewer`: General (ECC code reviewer),
  Security (security reviewer/security-review), Database/persistence (database
  reviewer/database-migrations/postgres-patterns), and Python (python reviewer/
  python-patterns). No UI/web-framework, release or other domain lanes were needed.
  General ran 165 focused tests/Ruff/mypy; Security ran 29 capture tests; Python
  ran 80 focused tests; Database inspected without rerunning checks.
- Fresh candidate-validation Task validated both findings below and deduplicated
  DB-1 into T075-1; no rejected candidates. Python returned verbatim:
  "No concrete task-relevant findings."
- Review-fix RED: `uv run pytest tests/unit/destination_capture_test.py -q -k
  'later_field or stable_sidecar' --tb=short --assert=plain` returned 3 failed,
  29 deselected: foreign model late-field identity, late WAL, and replace/restore.
  GREEN: the same command returned 3 passed, 29 deselected. Capture plus CLI tests
  then returned 83 passed. One strict mypy failure in the new connect-test wrapper
  was corrected with its explicit callable return type; the full chain restarted.
- Fresh exact recheck passed Ruff, strict mypy (89 source files), and 887 tests.
  The literal README sanitized native recipe also passed after both fixes.
- A fresh read-only disposition-verification `code-reviewer` Task verified both
  findings as RESOLVED, ran 83 focused tests and `git diff --check`, and reported
  verbatim: "New or unresolved task-relevant findings: None. No findings were deferred."
- One independent review/fix/recheck cycle completed. Taskrail verification follows
  the final exact validation chain; no apply/release/native GUI or Mobile claim is made.

## Implementation Notes

- Pin: `specs/v0.2.0.md#safe-update-application`; exactly this task cycle. The fetched
  accepted remote baseline was contained before writes, and the working checkout
  matched it. Candidate/priority/ownership validation passed before Taskrail start.
- Ownership path: focused `destination_capture.py` and the managed Typer subapp,
  with existing `destination_state` validation/adoption/journal contracts reused.
  No `cli.py` growth or default Anki dependency. Optional pinned generated protobuf
  definitions decode schema-18 model/templates/settings without opening a backend.
- Full SQLite tables/settings stay local; Personal Notes are not owned/importable
  values. Actual native note/card bindings, schedules, review rows and suspension
  observations remain distinct from eligibility and lifecycle ownership. Immutable
  whole-set source/object membership and destination/profile identity are explicit
  operator assertions, never visible-text matching or file/deck-name identity.
- Capture requires closed-copy/fresh no-edit attestations, rejects unsupported
  storage/layout/filtered decks/incomplete/ambiguous data, and does not approve apply.
  Adoption reviews each observed note's separated origins/keep overrides and
  atomically publishes a new journal without overwriting existing or pending state.
- README/docs provide a complete sanitized executable fixture and truthful limits;
  real proposals/handoff/observation envelopes remain operator-authored JSON.

### Validated independent findings and dispositions

T075-1 (General, medium; DB-1 is the validated duplicate):

> Capture can report a selected identity as absent when its note is under a different model and that model stores LatinitasID at a nonzero field ordinal.

Evidence: the original foreign-model branch checked only stored field zero
(`destination_capture.py:160-165`), despite capturing native field definitions.
The original wrong-model test only changed `mid` while leaving identity at field zero.
Disposition: fixed. Scan every foreign-model field for the exact selected portable
token, conservatively refusing a different model even if its fields were reordered
or renamed. This is rejection, never adoption from visible text. The late-field
regression failed before the fix and passed afterward.

T075-SEC-01 (Security, medium):

> The closed-backup and unchanged-file checks are not bound to a stable database file or sidecar-free interval. If another process replaces the path during capture and restores it before the final hash—or creates a WAL after the sidecar check—capture can return a snapshot of different or stale data while reporting the original artifact hash and fresh: true.

Evidence: the original pre-only sidecar check plus immutable pathname SQL connection
and main-file before/after hashes (`destination_capture.py:116-120,133-134`) did not
bind the acquired bytes to a stable file. Operator attestations are not a lock.
Disposition: fixed. Read one opened source into a private temporary byte copy;
derive the artifact digest and all SQL rows from those exact bytes; verify source
device/inode/size/mtime/ctime and visible sidecar absence at acquisition boundaries.
Reject replacement/restoration or a late sidecar, and remove the private copy on exit.
Both race regressions failed before the fix and passed afterward. Docs explicitly
retain the limitation that these boundary checks cannot prove no writer briefly ran.

No finding was deferred. Structural/migration, live collection integration, archive
acquisition and automatic proposal assembly remain outside this task's supported
scope; T-076–T-081 and release T-068 are not implemented in this cycle.
- 2026-10-07T23:38:54Z: verification pass
