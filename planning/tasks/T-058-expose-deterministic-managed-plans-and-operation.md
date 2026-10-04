---
id: T-058-expose-deterministic-managed-plans-and-operation
title: Expose deterministic managed plans and operation-bound approval
status: todo
priority: high
spec_ref: specs/v0.2.0.md#managed-update-plans
dependencies:
    - T-055-reconcile-managed-fields-and-destination-tag
    - T-056-plan-identity-preserving-card-retirement-and
    - T-057-reject-incompatible-managed-layouts-and-expose
    - T-053-render-safe-principal-part-comparisons-with
updated_at: "2026-10-04T23:05:46Z"
---

# T-058-expose-deterministic-managed-plans-and-operation Expose deterministic managed plans and operation-bound approval

## Description

Compose the reconciled state into inspectable machine-readable plans and a CLI review/approval workflow. Keep new command/domain implementation out of cli.py.

## Acceptance

- Classify notes as create/update/unchanged/conflict/retire with reasons, field/tag differences, proposed values, per-card effects, blocked operations, and transport capabilities.
- Record destination/snapshot fingerprint, baseline version, effective profile, schema/template binding, and a deterministic plan identity. Stable inputs yield stable output and no-op regeneration produces no unintended writes.
- Approval is explicit and bound to the selected operations and resolved plan; generation, profile confirmation, knowledge review, or a lifecycle tag is not apply approval. Modified plans/resolutions, including presentation changes affecting the approved payload, require renewed approval.
- Include structural and lifecycle proposals for inspection while reporting their selected-transport application as unsupported. Permit separately approved compatible content-only subsets only when retention/eligibility makes them safe and actual guard fields cannot indirectly add or remove cards.
- Cover create, unchanged, safe update, convergent/divergent edits, missing baseline, wrong destination, duplicates, incomplete and stale snapshots in sanitized CLI/plan fixtures.
- Verify approval against CSV's actual write footprint using an asymmetric fixture: approve field A, keep destination field B, leave another operation unapproved, and retain destination-only tags. Required unchanged columns carry resolved destination values; emitted columns/import mapping must not apply an unapproved value.
- No personal fields can enter approved write payloads. Preserve existing source-only export as the explicitly limited v0.1 path.

## Verification

Use red/green domain and CLI tests and the mandatory ruff, mypy, pytest -v chain.
