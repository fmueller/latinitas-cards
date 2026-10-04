---
id: T-046-preserve-untouched-exports-after-early-staging
title: Preserve untouched exports after early staging failure
status: todo
priority: high
spec_ref: specs/v0.1.1.md#authored-note-selection-and-preview
dependencies: []
updated_at: "2026-10-04T19:54:32Z"
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
