---
id: T-061-verify-managed-csv-import-safety-and-recovery-in
title: Verify managed CSV import safety and recovery in Anki
status: completed
priority: high
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies:
    - T-059-emit-approved-managed-csv-updates-and-reconcile
updated_at: "2026-10-05T05:35:16Z"
---

# T-061-verify-managed-csv-import-safety-and-recovery-in Verify managed CSV import safety and recovery in Anki

## Description

Exercise the supported native import/update path on a sanitized multi-card representative deck. This is an evidence gate, not a unit-test proxy for destination safety.

## Acceptance

- Record transport, Anki client/version, fixture setup, import settings, backup, and before/after offline destination observations.
- Verify content/tag updates and no-op reapplication produce no duplicate notes/cards, unchanged Personal Notes and destination-only tags, stable sibling bindings, and preserved scheduling/review histories for each card, including user-suspended cards.
- Separately test tag-only addition, removal, and removal of the final tag with every managed field unchanged; compare exact observed tag sets. Skipped intended changes remain pending or unsupported. Do not force an update by modifying unrelated content or metadata.
- Compare complete card and review-log tables and enumerate only expected note/content/tag changes; inspect sibling relationships and configured burying without implementing the scheduler.
- Exercise stale-plan rejection, changed GUI-handoff state, partial import reconciliation, backup/recovery, and retry, including import success before local recording fails, mixed field/tag outcomes, and backup restoration after recorded success. Verify approved subsets cannot overwrite kept fields or apply unapproved operations.
- Verify unsupported additions/retirement/reactivation/consolidation are refused and content-only plans cannot indirectly add or remove cards. Include a formerly absent slot becoming eligible during a sibling update and an enabled slot with a missing destination card; compare the actual card set.
- Verify separate-destination fresh starts leave original notes/cards/user data unchanged and disclose new scheduling.
- Run existing-recipe checks when the direct prerequisite is complete; form-parsing-exercises owns affected retests after its later changes. Bind evidence to tested content/setup and repeat affected checks for release-candidate changes.
- Publish exact supported capabilities and limitations. A failed or unavailable native run remains an open gate, not completed safety evidence. No actual user collection is required.

## Verification

Record reproducible sanitized observations and decisive table comparisons. Run the mandatory validation chain if fixes change code.

## Native evidence and workflow record

- Source guard: clean HEAD and origin/main both contained the accepted T-059
  predecessor; `taskrail validate` returned `state valid`, and `next --json`
  selected this task under `specs/v0.2.0.md#safe-update-application`. Exactly one
  task was started. No source-state transfer or manual STATE editing was needed.
- See `docs/managed-csv-native-verification.md` for the reproducible optional
  Anki 26.9.3 native API command, exact settings, fixture, source/script/report/
  backup hashes, complete comparisons, capabilities and limitations. All 16
  top-level native scenarios passed on 2026-10-05. Two notes/eight siblings/15
  synthetic reviews survived supported updates; no-op tables and journal were
  unchanged. Tag-only add/remove/final removal applied with every managed field
  unchanged. These are backend API observations, not Desktop GUI or mobile proof.
- Implementation is an evidence harness and documentation only. No production
  behavior, transport permission, slot contract or CSS boundary changed. No
  production RED/GREEN transition is claimed. Fixture setup errors (SQLite
  collation, explicit connection closing, canonical identity order and snapshot
  metadata shape) were diagnosed during native runs, not called product bugs.
- Initial checks and both later full chains passed: `uv run ruff check`,
  `uv run mypy` (84 source files), `uv run pytest -v` (825 passed). Final suite
  duration was 12.70s. Native checks are separate from that unit-test evidence.
- Mandatory simplification: a dedicated Task loaded `code-simplifier`, replaced
  callable Any types, removed repeated diff calculation and retained decoded
  actual burying config beside raw tables. Its new disposable native run passed;
  the owner inspected changes and repeated native/full checks.
- Independent read-only Task lanes loaded `code-reviewer`: General (acceptance/
  claims), Database (SQL/preservation), Security (synthetic data/path/freshness),
  Python (API/assertions). This is General plus three specialist lanes, the soft
  budget. No framework/ML/mobile lanes: no corresponding changed behavior or
  native mobile evidence. Each reviewer examined actual recorded observations.
  General, Security and Python each returned: "No concrete task-relevant
  findings." Task has no per-call model/effort control; medium routing was not
  asserted from an unavailable control.
- One fresh candidate-validation Task validated T-061-DB-1; there were no
  rejected or duplicate candidates. The finding below was fixed. A deliberate
  count-preserving native card-ID substitution made the strengthened assertion
  fail with `AssertionError: unexpected missing-card inventory` (exit 1). After
  removing that deliberate regression, the full native run passed, captured the
  exact original-minus-deleted card IDs and proved every captured destination
  table unchanged through rejected emission. No dummy edits force tag changes.

### Validated review finding (verbatim)

FINDING T-061-DB-1 — tests
Severity: low
Evidence: scripts/check-managed-anki.py:629-651 deletes one card, then the enabled-missing-card case asserts only that the resulting card count is seven. It does not assert the exact expected card-ID set or compare that set before and after the rejected emission. The report records the rejection as unsupported content-only card set change; docs/managed-csv-native-verification.md:77 describes this as an actual-set guard.
Finding: The enabled-slot/missing-card scenario does not verify the exact destination card set, so its evidence can pass with an unexpected card substitution that preserves the count.
Failure/impact: If the fixture or scenario later restores the deleted card but loses a different card, the count remains seven and the test does not catch that the expected missing-card inventory was not preserved.
Recommended direction: Assert that the captured card IDs equal the original set minus the deliberately deleted card, and that the set remains identical after the rejected emission.

Disposition: fixed with exact set/full-table assertions and native deliberate
regression RED → restored GREEN. Original finding line numbers describe the
reviewed pre-fix script. No findings were deferred. A fresh Task loaded
`code-reviewer` in disposition-verification mode, independently matched the final
script/report/backup hashes and before/after rejected-emission tables, and
returned "T-061-DB-1 — RESOLVED", "New or unresolved findings: None identified."
and "Ready for the scoped native gate." One review/disposition cycle was used;
verification/completion follows only after these gates and the final full chain.

## Implementation Notes

- 2026-10-05T05:35:16Z: verification pass
- 2026-10-05T05:35:16Z: Verified native backend safety for exact documented client/settings/fixture after workflow review and final chain; see docs/managed-csv-native-verification.md. T-060 owns affected recipe retests; T-062 remains separate.
