---
id: T-117-plan-annotation-patches-against-policy-and-approve
title: Plan annotation patches against policy and approve operations with declared actors
status: todo
priority: high
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies:
    - T-116-define-the-versioned-automation-policy-file-and
    - T-114-report-stale-orphaned-and-unmatched-annotations-in
    - T-115-record-corpus-claim-decisions-through-claim
updated_at: "2026-10-14T05:42:59Z"
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
