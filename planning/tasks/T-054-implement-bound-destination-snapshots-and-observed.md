---
id: T-054-implement-bound-destination-snapshots-and-observed
title: Implement bound destination snapshots and observed applied baselines
status: todo
priority: high
spec_ref: specs/v0.2.0.md#managed-update-plans
dependencies:
    - T-048-define-destination-snapshots-and-file-transport
updated_at: "2026-10-04T23:05:46Z"
---

# T-054-implement-bound-destination-snapshots-and-observed Implement bound destination snapshots and observed applied baselines

## Description

Implement the offline destination state and baseline contracts, keeping them distinct from source identity manifests and prior-export checkpoints.

## Acceptance

- Bind snapshots to the explicit destination, owned note type/schema/template layout, and managed note set; record completeness, available card/history evidence, and deterministic fingerprints.
- Missing or duplicate portable identities, unknown schemas, wrong destinations, incomplete snapshots, and absent baselines produce reconciliation/blocking outcomes, not empty destinations or safe overwrite. Never adopt notes by visible lemma alone.
- Provide explicit first-adoption/baseline initialization through observed destination state and reviewed ownership decisions. Persist managed values and separate source/configured tag contributions, user-owned overrides, and version/provenance. Personal Notes are excluded from baseline ownership.
- Persist observed per-operation application results and pending/unresolved operations. Support baseline advancement only for reconciled successful operations, never whole-plan advancement after partial failure.
- Prior exports or file existence cannot initialize or advance applied baselines implicitly.
- Sanitized fixtures cover create versus missing evidence, duplicate identity, wrong destination, unknown schema, successful observation, and partial results.
- Cover native import succeeding before result/baseline persistence fails or is interrupted, mixed observed field/tag outcomes within one note, and destination backup restoration after operations were recorded successful. Reacquire destination evidence and reconcile or invalidate inconsistent baselines; never blindly replay writes. The contract defines persistence and reconciliation granularity.

## Verification

Use red/green tests and the mandatory ruff, mypy, pytest -v chain.
