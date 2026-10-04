---
id: T-048-define-destination-snapshots-and-file-transport
title: Define destination snapshots and file transport capabilities
status: todo
priority: high
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-04T23:05:46Z"
---

# T-048-define-destination-snapshots-and-file-transport Define destination snapshots and file transport capabilities

## Description

Record the minimum destination snapshot, applied-baseline, and operation-by-transport contracts before implementing managed updates. The agreed initial scope is compatible content/tag CSV updates; structural additions, retirement, reactivation, and consolidation are planned but unsupported for application unless separately proven safe.

## Acceptance

- Specify destination binding to collection/export, Latinitas-owned note type/schema/template registry, and managed note set; include identities, managed values, tags, completeness, fingerprints, and freshness. Identify what card/suspension/review evidence is available and which operations its absence blocks.
- Define a practical offline snapshot acquisition and post-import observation path, distinguishing note-only CSV evidence from card/history evidence. Do not read or write a running collection.
- Publish a matrix for create/content/tag updates, card addition, retirement, reactivation, and migration across plain CSV, destination-aware CSV, and APKG. Unproved matching or structural capabilities are unsupported, not inferred from copied numeric IDs.
- Define explicit first-adoption reconciliation, successful observed import, partial application, retry, and baseline advancement. Prior-export checkpoints and generated files are not applied baselines.
- Document the fresh-snapshot/no-intervening-edits requirement for GUI import, the limit on enforcing that interval, and the circumstances requiring replanning or withholding preservation claims.
- Name sanitized native import checks required for each supported capability. No live integration or general template provisioning is brought forward from v0.6.0.

## Verification

Review the contract against every managed-update and lifecycle acceptance case in v0.2.0; record unresolved evidence explicitly rather than marking capabilities supported.
