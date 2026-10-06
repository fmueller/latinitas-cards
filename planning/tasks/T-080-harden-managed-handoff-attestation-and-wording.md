---
id: T-080-harden-managed-handoff-attestation-and-wording
title: Harden managed handoff attestation and output wording
status: todo
priority: low
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-06T20:17:05Z"
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

- Pending.

## Implementation Notes


