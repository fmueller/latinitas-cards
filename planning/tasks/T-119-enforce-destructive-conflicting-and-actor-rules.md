---
id: T-119-enforce-destructive-conflicting-and-actor-rules
title: Enforce destructive, conflicting and actor rules across plan, approve and apply
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#policy-gated-agent-automation
dependencies:
    - T-118-apply-planned-annotation-operations-atomically
updated_at: "2026-10-14T05:42:59Z"
---

# T-119-enforce-destructive-conflicting-and-actor-rules Enforce destructive, conflicting and actor rules across plan, approve and apply

## Description

Close the policy-gated flow of `specs/v0.3.0.md#policy-gated-agent-automation` across
`annotations plan`, `approve` and `apply`: decision-aware destructiveness, conflict and
actor rules end to end, atomic rejection of invalid batches, and the provenance chain in
`annotations inspect`. Fixed Contracts **Policy** ("Destructive and conflicting operations
are never `auto`: a matching `deny` still denies, otherwise they route to `review`";
`receipt_actor`) and **Apply-time safety** and **Exit codes** apply. T-117 persists full
plan-time safety facts, including accepted-or-withheld decisions from T-115; T-118
provides the multi-lock transaction. This task implements policy revalidation of pending
operations at apply time, comparing facts and routes, and proves it end to end. The `--claims
SIDECAR` review joins of `corpus show`, `corpus validate` and `annotations inspect` are
T-125. No command or option is added.

## Acceptance

- `annotations plan` marks a `supersede` or `retract` destructive when its target has an
  accepted or withheld version 2 decision in `<CORPUS>.claims.json` (the v0.2.1
  `sidecar_path` rule): the route is `review` with reason `destructive`, or `deny` when a
  `deny` rule matches; with no decision and no dependents the reason is absent. A decision
  matches when its `binding.annotation_id` equals the target, whether or not it is stale. A
  v1 file there exits 3 `unsupported_version`.
- For every not-yet-applied operation, apply recomputes validity, route and `safety`
  (`decision_status`, sorted dependent IDs, sorted conflicting IDs) against current corpus,
  current effective sidecar, entire embedded patch and current decisions, while holding
  T-118's corpus, sidecar and claims locks through replacement. Compare full facts and
  route to the plan, even if the route remains `review`: any changed fact/route or newly
  invalid pending operation rejects the whole batch before any append (4 `binding_drift`),
  requiring re-plan/re-approve. No claims digest is added; unrelated decisions do not
  invalidate the plan. Count patch-proposed records at plan and apply, deduplicated by ID.
- Adversarial decision tests: plan/approve a retract with no decision; subsequently record
  accepted, then in a separate case withheld, including stale bindings. The old receipt
  must fail apply with 4 and identical sidecar bytes, even though both old/new routes are
  `review`. Also change accepted to withheld; compare facts, not just destructiveness.
  Removing a decision must not silently downgrade an old route or reuse its receipt.
- Same-plan suffix tests add an unplanned dependent before a pending retract, and
  separately an unplanned conflict not represented in the patch: each causes 4/no append.
  Removing a dependent/conflict also causes drift rather than downgrading to `auto`.
  A foreign append still fails the base check. A dependent/conflict already represented
  by another patch operation must NOT cause false drift after its expected append;
  deduplicate by annotation ID and exclude an operation's own already-present record.
  Verify a patch-proposed dependent is counted before an earlier add can be appended in
  the same batch, so operation order cannot bypass safety checks.
- Newly invalid pending-operation tests: a same-plan suffix supersedes a segmentation
  used by a pending token annotation, and separately retracts a pending operation's
  target. Apply rejects the entire pending batch with 4/no byte change even if another
  pending add is valid; matching plan tags alone cannot authorize invalid anchors/targets.
- Controls: unrelated decision changes leave facts unchanged and apply succeeds;
  unchanged partial apply, later receipt and already-applied rerun still succeed without
  rerouting/reapplying completed operations. Absent claims means no decisions (lock
  creation allowed); unreadable, malformed and v1 claims files exit 3 with
  `unreadable`, `invalid_sidecar`, `unsupported_version`, not empty decisions. A held
  claims lock exits 4 `busy` without writes. A synchronized competing `claims record`
  cannot change decisions between apply revalidation and replacement; it retains the
  blocking lock protocol, not a new lock path.
- Conflicts end to end: two `add` operations with different values for one anchor and
  layer from one producer route `review` (`conflicting`) under the default policy and
  under a policy whose first rule is `auto` for that producer; the same pair under a
  `deny` rule routes `deny`; an `add` conflicting with a current sidecar record is
  `review`; two `add` operations identical except `created` (distinct `operation_id`s, same
  `annotation_id`) are not conflicting, both route normally and apply appends one record;
  byte-identical operations are `invalid` (`duplicate_operation`, T-117).
- Same codes regardless of who: one invalid patch (a stale `passage_hash` plus a
  morphology value without `tagset`) with producer kind `human` and with kind `agent`
  yields identical plan routes and reason lists; `annotations approve` of its invalid
  operation exits 3 `invalid_operation` for both `--actor human:ana` and `--actor
  agent:bot`; `annotations apply` of either plan exits 3 with identical diagnostics.
  Atomic batches: a plan with three valid `auto` adds and one invalid operation exits 1 at
  plan and 3 at apply, and the sidecar stays byte-identical (SHA-256 compared before and
  after); the same patch without the invalid operation appends all three.
- Actors: under the default policy `approve --actor agent:bot` exits 5 `actor_refused` and
  writes no receipt; a hand-built receipt naming `agent:bot` makes `apply` exit 5
  `actor_refused` with no byte change; with `receipt_actor: any` the same receipt applies.
  A `deny` operation exits 5 `denied` at approve and yields a `denied` warning with exit 0
  at apply, never appended.
- No route yields an accepted claim: after applying `auto` and receipted operations,
  `claims list CORPUS --annotations SIDECAR --json` shows every claim `review_status:
  "open"` and `<CORPUS>.claims.json` is unchanged or absent; an import test shows
  `annotation_plans.py`, `annotation_apply.py` and `policy.py` import no claim-recording
  function.
- Provenance: applying `tests/fixtures/annotations/fixture-morph.patch.json` (producer
  `{analyzer, fixture-morph, "1"}`) then `annotations inspect SIDECAR --corpus CORPUS
  --json` resolves each applied record to its `passage_id`, corpus `source.sha256`,
  producer, `input_hash` and the `plan_id` read from the plan file; `cltk`, `stanza` and
  `torch` are absent from `sys.modules`.
- A static test scans `tests/**/*.py` for `import`/`from` of `cltk`, `stanza`, `torch`,
  `openai`, `anthropic` and `ollama` and fails on any (negative control: a synthetic
  source).
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the plan routes and the before/after sidecar SHA-256 for each matrix case.

## Implementation Notes

- Implement the pending-operation safety recheck in `annotation_apply.py`, reusing the
  complete plan-time evaluator in `annotation_plans.py` and the v2 loader from T-115.
  Resolve decisions only from `<CORPUS>.claims.json`, matching `binding.annotation_id`
  even when stale. Keep the decision snapshot and all validation inside T-118's locks;
  do not read decisions unlocked and then reacquire locks just for replacement.
  Necessary edits to these modules, `annotation_state.py`, `annotation_sidecar.py`,
  existing shared I/O and command integration belong here, not a follow-up. Never import
  claim-recording functions, write decisions or authenticate declared actors.
- Put the matrix in `tests/unit/annotation_routing_test.py` with fixtures in
  `tests/fixtures/annotations/` and `tests/fixtures/policy/`; reuse the T-117 and T-118
  fixtures (`mixed.policy.json` and the `mixed*` patches) where possible.
- Update `docs/automation-policy.md` and the contract matrix with apply-time safety,
  fact drift, re-plan/re-approve and lock behaviour as well as destructive/conflicting
  definitions and actor rules. TDD starts with post-plan decision and same-plan-suffix
  RED cases, then the recheck, then partial-retry/no-op controls and concurrency.
  Size guard: focused apply/planning/state/I/O integration, schemas if required, docs
  and adversarial/routing tests. This is implementation work, not merely a matrix;
  required safety fixes may not be deferred because of the obsolete two-file guard.
