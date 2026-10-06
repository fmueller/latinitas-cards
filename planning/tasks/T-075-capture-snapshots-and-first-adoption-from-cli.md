---
id: T-075-capture-snapshots-and-first-adoption-from-cli
title: Capture destination snapshots and first adoption from the CLI
status: todo
priority: medium
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-06T20:17:05Z"
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

- Pending.

## Implementation Notes


