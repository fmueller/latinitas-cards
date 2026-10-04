---
id: T-046-preserve-untouched-exports-after-early-staging
title: Preserve untouched exports after early staging failure
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-selection-and-preview
dependencies: []
updated_at: "2026-10-04T20:10:54Z"
---

# T-046-preserve-untouched-exports-after-early-staging Preserve untouched exports after early staging failure

## Description

Fix the high-severity data-loss defect found in five-cycle test round 3.
When temporary CSV staging fails early with ENOSPC, existing later outputs
are deleted by rollback even though replacement never began. Failure in the
first stage loses form/QA outputs; failure in the second loses QA. The CLI
incorrectly reports that no output or committed state changed.

The shared commit boundary records destination existence only while staging;
unvisited artifacts retain a false default and rollback unlinks their originals.
Evidence: https://ampcode.com/threads/T-01a1086d-2eb5-725c-a468-a8b7d3ebe159

## Acceptance

- Strict red/green reproduces first- and second-stage ENOSPC with all three
  existing authored outputs and proves untouched later files remain exactly
  byte-identical after the fix, with input bytes unchanged and nonzero exit.
- Determine original existence for every artifact before staging can fail;
  cleanup never mistakes untouched existing files for newly created outputs.
- Test early staging failures at each position for existing, absent and mixed
  destinations, including non-text sentinel bytes and a stage-write failure.
  No new output or leaked staging file remains on complete rollback.
- Preserve existing replacement/interruption recovery and retained-backup
  semantics; error claims must match actual durable output state.
- Cover the shared boundary's legacy CSV, manifest and checkpoint consumers
  as applicable, not only authored per-kind export. Keep the fix narrow.
- Execute a real authored CLI fault-injection reproduction and verify the
  durable directory contents after failure, plus successful repeat export.
- Complete workflow-v3 review including persistence/data-loss invariants and
  disposition verification; `uv run ruff check`, `uv run mypy`, and
  `uv run pytest -v` pass.

## Verification Notes

- Record decisive regression results and verification timestamp; do not
  commit references to gitignored artifact paths.

## Implementation Notes

- Workflow 1: validated state and next-task JSON against accepted main base
  before starting; pinned specs/v0.1.1.md. Shared recovery must preserve every
  original destination and input, clean staging files, and report truthful errors.
- Workflow 2 RED: authored early-staging matrix returned 20 failed, 28 passed;
  stage 1 deleted form/QA, stage 2 deleted QA. Legacy writer returned 2 failed,
  4 passed, losing untouched manifest/checkpoint bytes. GREEN: snapshot original
  existence for all artifacts before staging; 152 focused tests passed.
- Matrix: all eight destination-presence subsets, all three staging positions,
  creation and write ENOSPC, binary sentinel bytes, whole-directory equality,
  unchanged source, successful export and deterministic repeat through real CLI.
  Legacy CSV/manifest/checkpoint coverage uses actual writer and binary originals.
- Workflow 3: initial exact Ruff/mypy/pytest chain passed with 626 tests after
  correcting fault-injection callback annotations found by mypy.
- Workflow 4: dedicated code-simplifier Task loaded its skill, inspected the diff,
  made no changes, and independently passed 152 focused tests.
- Workflow 5: parallel read-only code-reviewer Tasks loaded General, Security,
  Database (file-persistence recovery), and Python lanes with ECC guidance and
  companions. General, Database and Python: "No concrete task-relevant findings."
  No other framework/domain lane applies; three-specialist budget respected.
  Fresh candidate-validation Task validated Security S-1:

  FINDING S-1 — security
  Severity: medium
  Evidence: src/latinitas_cards/preview_export.py:590–596 records every destination’s exists() result before staging begins. On failure, :608–612 rolls back every artifact, and :842–851 unlinks a destination when existed_before is false and that path now exists.
  Finding: The early snapshot can cause rollback to delete a destination created concurrently after the snapshot.
  Failure/impact: If a destination is absent at the snapshot, another export creates it while this export is staging, and a later staging operation fails (for example, with ENOSPC), rollback treats the concurrent file as transaction-created and unlinks it. This can lose another export’s CSV or state.
  Recommended direction: Serialize commits for overlapping destinations, or otherwise ensure rollback removes only files this transaction installed; add a concurrent-creation rollback test.

- Workflow 6: S-1 fixed, no deferrals. RED concurrent-creation CLI test failed
  because form.csv disappeared. GREEN: staging failure skips destination rollback
  because no replacement has begun; temporary cleanup still runs. Focused suite
  passed 153 tests. Backup/replacement-phase recovery remains unchanged.
- Workflow 7: final exact Ruff/mypy/pytest chain passed (627 tests); fresh
  disposition-verification Task: "S-1 — RESOLVED" and
  "No concrete task-relevant findings." It independently passed 79 focused tests.
- Manual CLI check: actual app via CliRunner with mkstemp ENOSPC at calls 1/2
  exited 1 with the unchanged-state message; compared entire durable directory
  bytes and input, with no leaks. Successful export and repeat exited 0 with
  three byte-identical CSVs and unchanged source. Temporary sandbox removed.
- Workflow 8: verification and completion follow the reviewed final checks;
  Taskrail records the verification timestamp below. No release/deploy performed.
- 2026-10-04T20:10:54Z: verification pass
