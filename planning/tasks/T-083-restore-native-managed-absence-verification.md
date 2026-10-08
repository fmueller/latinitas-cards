---
id: T-083-restore-native-managed-absence-verification
title: Restore native managed verification after anchored-absence classification
status: completed
priority: medium
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-08T02:03:23Z"
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

### Executed implementation and review — 2026-10-08

- Pinned v0.2.0; fetched main contains accepted b5e2e8c and filing
  7021942. Taskrail validate/next selected only T-083 before start.
- Original documented native recipe on a fresh disposable directory exited 1
  at line 697: `AssertionError: unsafe operation was accepted`.
- Extracted the existing absence check unchanged into its owning harness
  helper; the focused test failed with that same assertion (RED). The helper
  now asserts anchored conflict/reconciliation, zero operations/card effects,
  and a verified empty approval with no targets/import rows. Fresh adoption
  of actual remaining members separately produces a nonempty create selection
  whose approval is refused (GREEN: one focused test passed).
- Optional backend imports are local to native entry points so default unit
  tests exercise the real helper without installing Anki. Production modules,
  classification and empty-selection behavior remain unchanged.
- Dedicated code-simplifier Task loaded its skill and made no changes; focused
  test and scoped ruff passed afterward.
- Separate read-only code-reviewer Tasks selected General, Python and Database
  (persistence/evidence) lanes. Each returned verbatim:
  "No concrete task-relevant findings."
  Security/framework lanes were omitted: no new trust boundary, transport,
  production persistence, or framework behavior. Candidate validation confirmed
  the empty candidate set; no rejected candidates, fixes or deferrals.
- Fresh disposition-verification Task returned verbatim:
  "No unresolved or newly identified task-relevant findings."
  It confirmed all 84 report-bound evidence hashes and reran the exact check
  chain successfully. One review/disposition cycle; no follow-up findings.
- Final exact chain: `uv run ruff check` passed; `uv run mypy` passed (90
  source files); `uv run pytest -v` passed (926 tests).
- Final documented native recipe using `anki==26.9.3` exited 0 on a new
  disposable directory and wrote report.json with 17 top-level cases. Both
  absence branches preserved every captured destination table. CSS mismatch,
  separate fresh start, and tag add/remove/final-remove all ran; the three
  tag-only outcomes were observed. Existing exact card/revlog/user-data and
  unresolved native-mismatch assertions remain intact.
- Executed script SHA-256:
  `2be32b1bd9617db35f03d577cd7fd34301c7fce2ec63b2129239f499d1a59f14`.
  Final report SHA-256:
  `ad142a82b4875333e857a2240afe3833513e095015cca5f85254ab3961e2d493`.
  Hashes bind executed content, not deterministic native IDs across reruns.
- Evidence is native Linux backend API only; no GUI/Mobile, migration,
  scheduler execution, or broad compatibility claim. No release action.
- 2026-10-08T02:03:23Z: verification pass
- 2026-10-08T02:03:23Z: Reviewed harness-only fix; mandatory chain and full 17-case native backend recipe pass. Evidence content hashes and review dispositions recorded in task notes.
