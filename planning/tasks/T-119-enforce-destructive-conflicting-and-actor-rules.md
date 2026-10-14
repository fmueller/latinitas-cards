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
`receipt_actor`) and **Exit codes** (`denied`, `actor_refused`) apply. T-117 computes
dependents and conflicts; this task adds the accepted-or-withheld decision fact from
`<CORPUS>.claims.json` (T-115) and proves the rules hold in every command. The `--claims
SIDECAR` review joins of `corpus show`, `corpus validate` and `annotations inspect` are
T-125. No command or option is added.

## Acceptance

- `annotations plan` marks a `supersede` or `retract` destructive when its target has an
  accepted or withheld version 2 decision in `<CORPUS>.claims.json` (the v0.2.1
  `sidecar_path` rule): the route is `review` with reason `destructive`, or `deny` when a
  `deny` rule matches; with no decision and no dependents the reason is absent. A decision
  matches when its `binding.annotation_id` equals the target, whether or not it is stale. A
  v1 file there exits 3 `unsupported_version`.
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

- Add a decision lookup to the routing-facts step in `annotation_plans.py` reading the v2
  file through `claim_sidecar` (read-only, no lock). The spec binds no claims digest into
  the plan, so a decision recorded after planning does not change an existing plan;
  document this limitation rather than adding a binding.
- Put the matrix in `tests/unit/annotation_routing_test.py` with fixtures in
  `tests/fixtures/annotations/` and `tests/fixtures/policy/`; reuse the T-117 and T-118
  fixtures (`mixed.policy.json` and the `mixed*` patches) where possible.
- Update `docs/automation-policy.md` with the destructive and conflicting definitions and
  the actor rule. Size guard: edits to `annotation_plans.py` and the routing tests; a
  missing capability elsewhere becomes a follow-up task.
