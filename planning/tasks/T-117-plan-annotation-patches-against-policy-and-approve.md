---
id: T-117-plan-annotation-patches-against-policy-and-approve
title: Plan annotation patches against policy and approve operations with declared actors
status: completed
priority: high
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies:
    - T-116-define-the-versioned-automation-policy-file-and
    - T-114-report-stale-orphaned-and-unmatched-annotations-in
    - T-115-record-corpus-claim-decisions-through-claim
updated_at: "2026-10-16T11:12:53Z"
---

# T-117-plan-annotation-patches-against-policy-and-approve Plan annotation patches against policy and approve operations with declared actors

## Description

Add the patch and plan halves of the annotation write flow in
`specs/v0.3.0.md#agent-ready-cli-contract`: `annotations plan PATCH --corpus CORPUS
--sidecar SIDECAR [--policy FILE] --output PLAN [--json]` and `annotations approve PLAN
--operation ID... --actor human:<id>|agent:<id> --review TEXT --output RECEIPT [--json]`
(Fixed Contracts **Commands**, **Versioned documents**: `latinitas-annotation-patch`,
`-plan`, `-receipt`; **Identity and hashing**: `operation_id`, `plan_id`). Approval
follows `managed approve` (`managed_plans.approve_plan`: explicit operation IDs, review
text, no approve-all). It uses the router (T-116), dependents (T-114) and `write_artifact`
(T-103). This task owns complete plan-time safety facts, including decisions from T-115.
Writing the sidecar is T-118; apply-time policy revalidation and the cross-command
adversarial matrix are implementation work owned by T-119.

## Acceptance

- `src/latinitas_cards/annotation_plans.py` defines the patch: `{format, version: 1,
  operations: [...]}` with `add` (`layer`, anchor, `value`, optional `alternatives`,
  `producer`, `confidence`, `created`), `supersede` (`target` annotation ID plus the same
  replacement fields) and `retract` (`target`, `producer`, optional `created`).
  `operation_id` is `sha256:` of canonical JSON of the operation. Floats, unknown keys,
  `version: 1.0` and unknown formats are rejected. A `supersede` operation's record sets
  `supersedes: <target>`. `latinitas-annotation-patch`, `latinitas-annotation-plan` and
  `latinitas-annotation-receipt` are registered in `json_schemas.SCHEMAS`; the shared drift
  test covers their `.v1.schema.json` files.
- `annotations plan` writes `{format: "latinitas-annotation-plan", version: 1, plan_id,
  base: {corpus_header, sidecar, policy, patch}, receipt_actor, patch, operations:
  [{operation_id, index, route, reasons, safety, records}]}` where `base` holds `sha256:` digests
  (`sidecar` null when the file is absent, `policy` null without `--policy`), `patch` is
  the canonical patch embedded for apply's recheck, `records` are the annotation or
  retraction records apply would append minus `plan` (apply sets it, avoiding a digest
  cycle) and minus `created` unless the patch gives it, and `plan_id` digests the plan
  without `plan_id`. `base.corpus_header` = `digest(header record)` (equal to `sha256` of
  the header line without `\n`); `base.sidecar` = `digest_bytes(file)` or null;
  `base.policy` = `policy_digest` or null; `base.patch` = `digest(canonical patch)`, i.e.
  the digest of the embedded `patch`.
- Operation-level `invalid` (reasons `unmatched`, `stale`, `orphaned` (anchor
  segmentation superseded), `input_hash`, `invalid_value` such as morphology without
  `tagset`, `span_out_of_range`, `unknown_segmentation`, `unknown_target`,
  `target_inactive`, `duplicate_operation`; the closed reason set in
  `docs/automation-policy.md` is extended with every `invalid` reason) still writes PLAN
  and exits 1; otherwise exit 0, also with `review` or `deny`. Whole-input rejection exits
  3 and writes nothing: unreadable or non-JSON patch, wrong format, higher version, a
  policy with `threshold` or `auto` for `supersede` (same codes as `policy validate`), a
  corpus with findings (one `invalid_corpus` error whose message names the first finding
  code, plus the findings) or an unsupported/invalid sidecar (`unsupported_format`,
  `unsupported_version`, `invalid_sidecar`).
- Routing calls `policy.route` with `destructive` true for a `supersede`/`retract` whose
  target has dependents (`annotation_state.dependents`) or an accepted/withheld v2
  decision matching `binding.annotation_id`, even if stale, and `conflicting` true for two
  operations in the patch, or an operation and a current sidecar record, giving different
  values for one anchor and layer from the same producer (`kind`, `name`, `version`);
  identical values are not conflicting. An `add` already present in the sidecar is routed
  normally (apply skips it).
- Persist per-operation `safety: {decision_status: accepted|withheld|null, dependents:
  [sorted annotation IDs], conflicts: [sorted annotation IDs]}`; decision status is null
  and dependents empty for add. Compute dependents/conflicts against the effective
  sidecar and the entire embedded patch, deduplicated by annotation ID; a proposed
  dependent/conflict must count before any append. Own already-present records do not
  conflict with themselves. Test a patch-proposed dependent on a retract target and
  same-ID proposals with differing `created`: sorted facts and routes must not change
  when an expected proposal later appears in a same-plan suffix.
- Read decisions only from `<CORPUS>.claims.json`, never a new option or claims digest
  in `base`. Absence means no decisions; unreadable/malformed/v1 files exit 3
  (`unreadable`/`invalid_sidecar`/`unsupported_version`), not an empty set. Tests cover
  accepted, withheld and stale bindings, plus absence and each rejected file class.
  These are contract-group exits, distinct from the legacy `claims` exit-1 failures.
- Output rules: an identical existing PLAN or RECEIPT is a no-op (exit 0); different bytes
  exit 3 `exists`; an output aliasing PATCH, CORPUS, SIDECAR, the policy,
  `<CORPUS>.claims.json`, a manifest or any `.lock` (direct, symlink, hard link) exits 3
  (code `alias`); a held `<PLAN>.lock`/`<RECEIPT>.lock` exits 4 `busy`.
- `annotations approve` recomputes `plan_id` (tampered plan: exit 3 `invalid_plan`),
  writes `{format:
  "latinitas-annotation-receipt", version: 1, plan_id, operations: [sorted IDs], actor,
  review}`, and exits 3 for an unknown ID (`unknown_operation`) or an `invalid` one
  (`invalid_operation`), 5 for a `deny` one (`denied`) and 5 `actor_refused` for `--actor
  agent:<id>` when the plan's `receipt_actor` is `human`; mixed failures follow 3 > 5;
  missing `--operation` is a Click error (exit 2, not a matrix row). Nothing is written on
  nonzero exit. Matrix rows: `annotations plan: 0, 1, 3, 4`, `annotations approve: 0, 3, 4,
  5`.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the committed fixture plan's `plan_id`.

## Implementation Notes

- New `annotation_plans.py` (patch/plan/receipt models, classification, `plan_id`) and
  callbacks in `commands/annotations.py`; reuse `annotation_sidecar.py` records and IDs,
  the corpus reader `corpus_io` (T-104), `policy.load_policy`/`route`/`policy_digest`, and
  `atomic_io.write_artifact`/`reject_aliases`. Plans store no paths.
- Keep safety-fact evaluation reusable by T-119's apply revalidation, with a current
  corpus, effective sidecar, whole patch and decision snapshot supplied by the caller.
  Reuse T-115's v2 loader read-only; no claim-recording calls. Update the plan schema,
  fixtures and docs with the required safety fields now, not as a later schema retrofit.
- Fixtures under `tests/fixtures/annotations/`: `fixture-morph.patch.json` (auto adds),
  `mixed.patch.json` (one auto add, one supersede, one deny-matched add, one invalid op),
  `mixed-valid.patch.json` (the same without its invalid operation) and
  `mixed.base.annotations.jsonl` (the sidecar holding the record the `supersede` targets);
  and `tests/fixtures/policy/mixed.policy.json` (a `deny` rule for the deny-matched add's
  producer name, `receipt_actor: human`), reused by T-118 and T-119.
- An empty `--review` follows `managed_plans` (`_text`): reject it, and a malformed
  `--actor` value, with exit 3 `invalid_input`. Approving an `auto` operation is not fixed
  by the spec: accept an `auto` ID as a no-op selection. Document all three in
  `docs/agent-cli-contract.md`.
- TDD order: patch model, IDs, classification, routing facts, output rules, approve. Size
  guard: one module plus two callbacks, v2 loader reuse and required plan schema/fact
  tests. Complete plan-time facts belong here; do not defer them because of size guards.

## Workflow-v3 execution evidence

- Scope: exactly T-117, accepted origin/main `48b29a5f0f2f624580abc025105e47e2f1d69585`;
  pinned spec SHA256 `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
  Fetch, `mise exec -- taskrail validate` and `next --json` confirmed the base and
  selector before start (2026-10-16T10:49:14Z). The earlier STATE timestamp-only
  drift had no current task/owner; no product changes were transferred.
- Reading: acceptance-only contract, fixed contracts lines 349–405 and 485–544,
  plus bounded fixture/approval constraints relevant to implementation. Full task
  and full spec were not read. IO Delegate declined bulk dispatch with
  `subscription_approval`/`not_dispatched`; owner refined the request to bounded
  native acceptance inspection. No provider/settings changes or equivalent bulk
  reconstruction were made.
- TDD: real CLI `test_plan_identity_and_explicit_approval` initially failed with
  `No such command 'plan'`; first implementation passed five tests. The v2
  accepted/withheld + stale/current matrix then failed four tests (exit 70),
  fixed by reading `entry.decision.status` from the existing v2 loader model.
  Reserved manifest/lock tests failed seven cases (`exists` instead of `alias`),
  then passed after alias protection. Approval-input/no-op/required-fixture tests
  failed five cases before their implementation; renamed protected receipt
  destination tests failed six cases before document identity protection.
- Kept contract checks cover canonical IDs/no paths, full-patch dependent/conflict
  facts, same-ID proposals with different `created`, same-plan suffix stability,
  target lifecycle, strict patch input, no-output refusals, explicit review and
  declared actors, mixed exit precedence, immutable output and held locks.
  Fixture capture remains temporary until all shared destination guards pass;
  tests cover symlink, hardlink and no-force overwrite safety. T-127 exporter,
  T-128 overhead, T-038 release and T-118/T-119 apply remain unchanged.
- Initial broad run exposed two float-test setup errors (the test serializer
  rejected floats before the CLI) and the old command-registration expectation;
  corrected the setup to serialize untrusted input and extended the registration
  set. Focused contracts/schema/matrix/generator checks: 135 passed (one existing
  Pydantic serializer warning). No failing run is treated as a passing gate.
- Initial exact chain subsequently passed: ruff, mypy (159 source files),
  pytest (2315 passed, one existing warning, 95.33 seconds).
- Dedicated code-simplifier Task loaded its personal skill. Its initial source
  read was denied and made no edits; a genuinely narrowed 58-line control-flow
  question succeeded. Accepted explicit decision-status and layer-selection
  branches instead of nested conditionals; behavior/contracts unchanged.
  Post-simplification: 54 focused tests, ruff and mypy passed.
- Four separate parallel code-reviewer Tasks loaded their personal skill and
  mapped ECC references: General; Security with security-review; Database with
  postgres-patterns/database-migrations (filesystem locking/integrity only);
  Python with python-patterns. These cover correctness, trust boundaries,
  file persistence and Pydantic contracts within the three-specialist budget.
  FastAPI omitted: no web server/framework API; Pydantic covered by Python.
  No other language/framework/domain lane applies. General, Database and Python
  each returned verbatim: "No concrete task-relevant findings."
- Security candidate F-1, retained verbatim below, was rejected by a fresh
  code-reviewer candidate-validation Task. The specified plain digest provides
  consistency, not authenticity; approve has no trusted policy/source paths,
  signing keys or trust store. Recomputing a changed body's digest is not a
  T-117 contract violation. No validated findings, fixes or deferrals resulted.

> FINDING F-1 — security
> Severity: medium
> Evidence: `src/latinitas_cards/annotation_plans.py:119–128` verifies that `plan_id` matches the document’s contents and that operation IDs match the patch, but does not recompute routing. `src/latinitas_cards/annotation_plans.py:440–443` trusts the stored route when deciding whether to reject an approval.
> Finding: Approval can accept a policy-denied operation if someone edits the plan’s route to `auto` and recomputes `plan_id`; the digest detects edits only when it is left stale.
> Failure/impact: Starting with a denied operation, change its route to `auto`, recompute the document digest as `plan_id`, then approve its unchanged operation ID. The plan passes validation and a receipt is written, bypassing the policy-deny gate for that approval artifact.
> Recommended direction: Before approving, verify the selected operation’s route against trusted inputs or otherwise establish that the plan is an authentic plan result; add a regression test for a modified-and-rehashed denied plan.

- Additional deliberate-regression validation: removing proposed records from
  effective facts failed both whole-patch/suffix tests (2 failed); restoring
  production yielded 2 passed. Removing plan-ID verification failed the tamper
  test (1 failed), restoration yielded 1 passed. Bypassing approval's diagnostic
  guard failed deny/actor/mixed-precedence tests (3 failed, 1 passed). All temporary
  regressions were restored, not committed. These discriminate plausible unsafe
  implementations rather than only exercising successful paths.
- Real CLI proof: `uv run latinitas-cards annotations plan` on fixture-morph patch
  returned exit 0, route auto, plan ID
  `sha256:b4ea3a44ff3364a31f560fe1ff7d6767b26356139c1595de9d2c7c83e7e3db45`;
  `annotations approve` selected operation
  `sha256:cf8775bf215b128ff2acff35ef700f3486140cc5e76ba6aabac6c862573e16ce`
  with human:manual/review text and returned receipt exit 0. Before/after SHA256
  of the sidecar matched. Temporary manual artifacts were removed.
- Fresh disposition-verification Task loaded code-reviewer plus Security/Python
  guidance and companions. Result verbatim: "No concrete task-relevant findings."
  Confirmed F-1 rejected, no disposition required; focused approval/identity
  verification 5 passed. One review/candidate/disposition-verification cycle;
  no unresolved findings or new task follow-ups.
- Final exact `uv run ruff check` → `uv run mypy` → `uv run pytest -v` passed:
  all checks passed, 159 source files, 2315 tests (91.10 seconds), one existing
  Pydantic warning. `mise run check` passed independently: format 371 files,
  ruff/mypy, 2315 tests (91.21 seconds), commit/push/author guards, differential
  and full mutation-report guards and mutation configuration checks.
  `git diff --check` clean; fresh origin fetch and spec SHA256 still matched
  accepted base/pin before lifecycle finalization. Hooks installed through
  `mise exec -- lefthook install`; maintainer identity confirmed, no trailers.
- 2026-10-16T11:12:53Z: verification pass
- Taskrail verify/complete both succeeded at 2026-10-16T11:12:53Z, after all
  review and verification gates. State valid, current task cleared. Next selector
  is T-118-apply-planned-annotation-operations-atomically; not started in this cycle.
  No follow-up task was needed. T-127 remains todo/exporter unfixed, T-128 remains
  unstarted v0.8.0 work, and T-038 remains blocked/maintainer-only.
