---
id: T-054-implement-bound-destination-snapshots-and-observed
title: Implement bound destination snapshots and observed applied baselines
status: completed
priority: high
spec_ref: specs/v0.2.0.md#managed-update-plans
dependencies:
    - T-048-define-destination-snapshots-and-file-transport
updated_at: "2026-10-05T02:35:23Z"
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

## Implementation evidence

- Added the offline `destination_state` domain API and its documented version-1
  evidence/journal formats. No CLI, running collection, native acquisition/import,
  scheduling proof or structural transport was added.
- Snapshot binding includes collection/profile, exact authoritative field ownership,
  schema/registry and actual template/CSS digests, immutable source/object membership,
  completeness/freshness assertions, counts and available full card/history evidence.
  Deterministic fingerprints keep acquisition/artifact references separate.
- Adoption requires explicit approval and every observed ownership decision. Atomic
  journal persistence keeps confirmed note anchors/receipts together; unresolved
  operations retain anchors and whole-plan status remains separate. Explicit reviewed
  recovery accepts observed state, abandons pending effects and retains historical receipts.
- RED: focused tests initially failed because the module did not exist. GREEN: 23
  synthetic tests after adoption/recovery coverage. Final focused suite: 33 passed.
- Dedicated simplifier loaded code-simplifier and reused one decoded observation payload;
  focused tests passed afterward. Independent General, Database/persistence and Security
  input-boundary lanes loaded code-reviewer and their mapped guidance. Separate Python
  lane omitted initially because General plus static checks covered language concerns;
  final disposition checks used Python guidance. No SQL/framework/ML/native lanes applied.
- Fresh candidate validation accepted all three initial findings without duplicates or
  rejected candidates. Fresh disposition verification confirmed their fixes and identified
  one construction bypass, validated and fixed in the second and final cycle.

### Review findings and dispositions

- G-1: "`reconcile` can report a baseline as consistent after collection deck options
  change or are restored from a backup, even though the API treats unchanged deck options
  as part of confirmed application evidence." Fixed: persist baseline deck options and
  compare during reconciliation; changed-options RED (DID NOT RAISE) then GREEN.
- DB-1: "`_state` accepts internally inconsistent journals where a plan is marked
  `complete` while one or more operations remain `pending` or `unresolved`, allowing
  `begin_observation` to bypass the pending-effects guard." Fixed: validate aggregate
  statuses at all boundaries; RED (DID NOT RAISE) then GREEN. Also proved/fixed empty
  attempts creating an inconsistent pending journal with a RED/GREEN regression.
- S-1: "`read_snapshot` can treat an identity excluded from the export as proven absent,
  despite the contract allowing absence only when complete evidence establishes it."
  Fixed: conservative v1 rejects all exclusions; identity/query exclusions RED (three
  DID NOT RAISE failures) then GREEN; documented this practical limit.
- NEW-1: "Public construction of `DestinationSnapshot` bypasses snapshot validation,
  allowing stale or excluded evidence to pass reconciliation and start a pending
  observation." Fixed: constructor uses shared validation/canonicalization; four direct
  construction RED (DID NOT RAISE) then GREEN.
- Final fresh reviewer: "New task-relevant findings: No concrete findings from the fixes
  reviewed." All four dispositions resolved; none deferred; two review/fix cycles total.
- Exact final gates: `uv run ruff check` (All checks passed), `uv run mypy` (74 files,
  no issues), `uv run pytest -v` (724 passed). Final reviewer independently reran the same
  chain successfully. `git diff --check` passed.
- Fixtures cover complete absence versus missing evidence, duplicates/wrong binding/schema,
  partial A/B and mixed field/tag observations, unknown/changed preservation evidence,
  interrupted persistence, restored backups, keep ownership and foreign prior-export data.
  Native proof and actual apply remain later scoped tasks; persistence is single-writer.

## Implementation Notes

- 2026-10-05T02:35:23Z: verification pass
- 2026-10-05T02:35:23Z: Implemented offline evidence/adoption/atomic observed journal and explicit recovery. All gates and independent disposition verification passed; no native apply claim.
