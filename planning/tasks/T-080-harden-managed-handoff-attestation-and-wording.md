---
id: T-080-harden-managed-handoff-attestation-and-wording
title: Harden managed handoff attestation and output wording
status: completed
priority: low
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-08T01:14:48Z"
---

# T-080-harden-managed-handoff-attestation-and-wording Harden managed handoff attestation and output wording

## Description

Found by the T-069 review. Backup is any non-empty operator-attested file and snapshot freshness
is self-asserted with no age bound; NUL/control characters in managed field values reach the
CSV unquoted; CLI output still prints `native_safety: unverified; T-061 gate required` and plans
report `application_implemented: false` although `emit` exists. Not release-blocking.

## Acceptance

- Docs and CLI output state the operator-attested backup/freshness trust model plainly.
- Control characters in managed field values are rejected or safely handled, with tests.
- User-facing managed output drops internal task references and explains the offline handoff.

## Verification Notes

- RED: 15 targeted failures (14 proposal/destination active-control cases did not
  raise; offline application capability still false). GREEN: all pass after shared
  managed-field validation/reporting changes. CLI NUL refusal assertions also failed
  under a deliberate validator regression, then passed with validation restored.
- Final exact chain: `uv run ruff check` passed; `uv run mypy` passed (89 files);
  `uv run pytest -v` passed (924 tests). An intermediate targeted failure in the
  terminal escaping fixture was resolved by retaining C1 coverage in metadata and
  permitted Unicode coverage in fields; full chain restarted successfully.
- Dedicated code-simplifier pass: no edits. Separate independent code-reviewer lanes:
  General, Security, Database/persistence, Python. All returned "No concrete
  task-relevant findings." Candidate validation and fresh disposition verification
  confirmed the empty set; no fixes deferred. One review cycle.
- Installed CLI smoke: supported emission reported backup/freshness/no-edit operator
  attestations, manual native import and observation, no automated apply or age/lock
  enforcement; emitted effects pending with unchanged anchors. NUL snapshot failed
  with `unsafe control U+0000 in managed field Meaning`, exit 1, no unsafe CSV and
  byte-identical journal. No new native probe or expanded compatibility claim.

## Implementation Notes

- Shared managed-field validation rejects C0/C1/DEL except tab/CR/LF without changing
  accepted text. Asymmetric Unicode, quoted multiline/CRLF/tab CSV round-trip and
  both unsafe proposal/destination boundaries are covered. Personal Notes remain
  outside managed writes; terminal escaping and canonical metadata remain unchanged.
- Reports/help/docs explain nonempty backup/hash is not native recoverability proof;
  closed capture boundary checks are not a collection lock. Snapshot freshness and
  edit/review/sync-free interval are operator attestations, not TTL enforcement.
- Plan capability now identifies implemented offline CSV emission/manual handoff/
  observed-result reconciliation, explicitly not automated collection apply. Existing
  backend/Desktop 26.09.3 fixture/settings limits, selected-operation/full-tag footprint,
  provenance, backup aliases, races and lifecycle restrictions remain unchanged.
- 2026-10-08T01:14:48Z: verification pass
- 2026-10-08T01:14:48Z: One reviewed T080 cycle complete; operator attestations explicit, unsafe managed controls rejected, offline handoff truthful. No expanded native safety claim.
