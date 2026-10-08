---
id: T-083-restore-native-managed-absence-verification
title: Restore native managed verification after anchored-absence classification
status: todo
priority: medium
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-08T01:49:03Z"
---

# T-083-restore-native-managed-absence-verification Restore native managed verification after anchored-absence classification

## Description

Confirmed in restarted adversarial cycle round 2 on 2026-10-08 at accepted
revision 071fd0e994ec266009cd24f843abf8220e8a2056, pinned to v0.2.0.
The documented native verification recipe exits 1 after T-078 correctly changed
previously anchored destination absence from `create` to a blocked conflict.
This is a verification-harness regression, not evidence of unsafe application.

Deterministic reproduction (new disposable output path; no user collection):

```sh
work=$(mktemp -d)
uv run --with anki==26.9.3 python scripts/check-managed-anki.py \
  --output-dir "$work/evidence"
```

Expected: the documented native gate completes successfully, verifies anchored
absence is a blocked conflict with no write targets, separately verifies genuinely
never-anchored creation remains unsupported, and reaches the later CSS drift,
fresh-start and tag-only branches with their existing exact-table assertions.

Actual: exit 1 at `scripts/check-managed-anki.py:697`, calling
`reject(lambda: approve_plan(unsafe, ops, "must refuse"))`; `reject()` line 288
raises `AssertionError: unsafe operation was accepted`. No final `report.json`
or success marker is produced, and the subsequent branches do not run.

Independent installed CLI plan/approve reproduction against the script's actual
closed native before/absent snapshots confirms the correct product result:

```json
{
  "classification": "conflict",
  "blocked": ["previously anchored note absent; destination reconciliation required"],
  "operations": [],
  "approval_exit": 0,
  "targets": {},
  "import_rows": [],
  "selected_operations": []
}
```

Root: the absent-note branch at lines 688-698 still collects all operation IDs
and expects unsupported-create rejection. `managed_plans.compose_plan()` now
correctly emits no operations for anchored absence. `approve_plan()` accepts an
empty selection as an empty receipt; it authorizes no write. Do not regress the
T-078 classification or the ordinary empty-selection contract to satisfy the
obsolete harness assertion.

This executable is advertised in `docs/managed-csv-native-verification.md` and
`docs/contextual-form-parsing.md`. Safe Update Application requires reproducible
actual-transport preservation evidence. Deduplication: T-078 fixed production
classification and unit coverage but did not update this native gate; T-075's
separate capture walkthrough passes, and T-082's no-write resolutions also pass.
No existing open task covers this downstream verification regression.

## Acceptance

- The anchored-absence native branch asserts conflict, explicit reconciliation reason, no operations/card effects, and no import/write footprint; the destination remains unchanged by planning/approval.
- A separate genuinely never-anchored absent-member branch still proves creation is unsupported at approval, without weakening the baseline distinction or selecting an empty operation list as the refusal test.
- The full documented command above exits 0 on a new disposable output directory and writes its final report, reaching CSS mismatch, fresh-start and all tag-only branches. Preserve exact cards/history/user-data assertions and truthful unresolved native-import mismatches.
- Add a focused regression that fails when the harness confuses anchored absence with unsupported creation. Run the mandatory ruff/mypy/pytest chain and the native recipe; do not claim Desktop/Mobile, migration, scheduler execution, or broad compatibility.

## Verification Notes

- 2026-10-08: reproduced native command exit 1 and independently confirmed the zero-write CLI receipt. No product fix applied in this filing.
- Same pinned revision passes `uv run ruff check`, `uv run mypy` (89 source files), and `uv run pytest -v` (925 tests); the current unit suite therefore does not detect this executable-gate regression.
- Fresh round-2 capture/adoption native sequence passed full table comparisons for two notes, eight asymmetric cards and fifteen review rows. Six fresh offline no-write/mixed-convergence CLI sequences passed 66 assertions across 54 installed invocations. These successes do not erase this gate failure.

## Implementation Notes

Keep the fix in the native verification harness and focused regression coverage.
No production behavior change, release action, dependency change, task lifecycle
transition, or new transport capability is requested by this finding.
