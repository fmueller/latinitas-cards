---
id: T-036-preserve-csv-and-manifest-recovery-on-interruption
title: Preserve CSV and manifest recovery on interruption
status: todo
priority: high
spec_ref: specs/v0.1.0.md#interrupt-safe-csv-and-manifest-export
dependencies: []
updated_at: "2026-09-27T09:01:50Z"
---

# T-036-preserve-csv-and-manifest-recovery-on-interruption Preserve CSV and manifest recovery on interruption

## Description

Fix catchable interruption recovery in preview_export.py's CSV/manifest transaction.
The architecture review reproduced KeyboardInterrupt after backups were moved: both
destinations and all backups disappeared because rollback caught only OSError while
finally cleanup still ran. This is distinct from hard-crash pair atomicity.

## Acceptance

- Reproduce using a synthetic source CSV and existing output/identity manifest, injecting
  KeyboardInterrupt before output replacement after backup moves. Preserve prior bytes
  and authoritative identity assignments, or retain discoverable recoverable backups
  with clear recovery diagnostics if restoration fails; source remains unchanged.
- Propagate cancellation after recovery; do not swallow interrupts. Remove backups only
  after confirmed commit or recovery, not unconditional finally cleanup.
- Inject interruption around every destructive move/replacement, with both destinations
  present, both absent, and each mixed presence state; include partial commit, failed
  restoration and interruption during recovery. Never destroy the last recoverable copy.
- Retain ordinary OSError rollback coverage. Assert file contents, source-ID assignments,
  destination/backup presence and cancellation propagation rather than just exceptions.
- Limit implementation to existing export transaction/cleanup boundaries; no database,
  transaction framework or stronger power-loss/hard-crash atomicity promise.

## Verification Notes

- Add red/green regressions based on the reproduced failure and boundary matrix above.
  Run the mandatory ruff/mypy/pytest chain after implementation; record actual evidence.
- This task is unimplemented planning. T-035 depends on it and T-014 remains blocked.

## Implementation Notes
- Independent of the note-model change; preserve T-026/T-027/T-028 completed history.
