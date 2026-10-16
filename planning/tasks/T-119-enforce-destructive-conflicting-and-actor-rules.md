---
id: T-119-enforce-destructive-conflicting-and-actor-rules
title: Enforce destructive, conflicting and actor rules across plan, approve and apply
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#policy-gated-agent-automation
dependencies:
    - T-118-apply-planned-annotation-operations-atomically
updated_at: "2026-10-16T12:29:59Z"
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

## Execution Evidence

- Ownership: fetched origin/main and HEAD both matched the accepted base
  `4366d5bd6dde253ee12f8bd4f82316b5da8d6ee6`; spec SHA-256 matched
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
  `taskrail validate` passed and `next --json` selected exactly T-119 before
  `start` at 2026-10-16T12:00:56Z. All Taskrail calls use `mise exec -- taskrail`.
- Strict TDD: first decision/same-plan-suffix run produced four failures
  (`assert 0 == 4`: unsafe success instead of binding drift). Locked evaluator
  integration made those tests plus the existing apply suite pass (62 tests).
  An additional completed-patch segmentation revival test reproduced unsafe
  success; the evaluator lifecycle correction turned it green. Final focused
  planning/apply/routing suite: 152 passed.
- Matrix: 40 routing cases cover fresh/stale accepted/withheld decisions,
  changed/removed decisions, unrelated decisions, added/removed dependents and
  conflicts, expected patch-proposed suffixes, invalid anchors/targets, actor
  declarations, deny precedence, created-only duplicates, invalid batch atomicity,
  provenance, no accepted claims and static optional-provider import rejection.
  Each matrix apply logs routes and before/after sidecar SHA-256; rejected
  batches assert identical hashes. Existing apply tests also cover foreign
  suffixes, all three busy locks, the exact logical symlinked claims lock and
  a synchronized blocking claims writer held through replacement.
- Dedicated `code-simplifier` Task loaded its personal skill, reused one current
  records collection in apply and one AST parse in the import scan; 152 focused
  tests passed. Parent inspected the diff and reran routing/apply: 98 passed.
- Separate parallel read-only `code-reviewer` Tasks selected General, Security
  and Database (persistence/concurrency) lanes. General loaded its ECC reviewer;
  Security loaded its ECC reviewer, security-review, linked cloud-security
  resource and common security rules; Database loaded its ECC reviewer,
  postgres-patterns and database-migrations (SQL guidance not applicable).
  Python-specific review was not a separate lane: General covers this small
  typed-Python integration; no framework, schema or SQL changes require more
  lanes. Two specialist lanes stay within the default budget.
- All three returned verbatim: "No concrete task-relevant findings."
  A fresh candidate-validation `code-reviewer` Task inspected the actual diff,
  loaded General/Python/Security/Database references and applicable companions,
  and independently ran 152 focused tests. Its conclusion was also:
  "No concrete task-relevant findings." Candidate IDs accepted/rejected: none;
  dispositions/finding-fix TDD: none required. No candidates were suppressed or
  deferred. This is one review cycle; final disposition verification follows
  the final gates before lifecycle completion.
- Scope: no new command/option, authentication, claims digest/decision writes,
  provider changes, spec/gate/dependency/priority edits or second task. T-127
  schema exporter, future v0.8.0 T-128 memory work and blocked T-038 release
  publication remain unchanged and excluded. No in-scope follow-up is needed.
- Self-verification strengthened the inherited competing-writer test with a
  subprocess handshake at the actual blocking claims-lock syscall, avoiding a
  false positive from slow CLI startup. A process-local deliberate regression
  omitting only apply's claims lock failed with `DID NOT RAISE TimeoutExpired`;
  the unchanged correct implementation passed the strengthened test (1 passed).
  No production lock behavior or on-disk source was changed by that experiment.
- The first `mise run check` caught formatting in the simplifier's AST helper.
  Ruff formatting corrected it; the full exact chain and repository gate were
  rerun rather than waiving the formatting failure. Earlier full-chain counts
  were 2404 before final matrix expansion and 2418 after expansion.
- Isolated accepted-base apply replay (no on-disk code edits) reran the expanded
  decision/suffix regressions: 6 failed, 34 deselected, all returning unsafe exit
  0. For the decision cases, sidecar SHA-256 changed from
  `cb822ae209d8905721a582b3e271582f8da76f9226a1c017ccf167afb617ce4e`
  to `1fe5dcdddd4532544c101b8c0773ed23b55ba1d7949fbc66e8262545c33cd5b2`.
  Corrected CLI envelopes return exit 4 `binding_drift` and retain the original
  hash. The current matrix run reports 40 passed, including partial/no-op controls.
- Final exact chain after all code/test changes: `uv run ruff check` passed;
  `uv run mypy` reported no issues in 162 source files; `uv run pytest -v`
  reported 2418 passed, one existing Pydantic serialization warning, in 127.86s.
  `mise run check` passed (2418 tests, format check and all five shell guard
  suites); a separate format check reported 374 files already formatted.
  Gates were confirmed at 2026-10-16T12:25:09Z.
- First fresh disposition-verification Task loaded `code-reviewer` and
  Python/security/database references and inspected the new lock handshake.
  It returned no implementation issue but raised this candidate verbatim:

  > FINDING T-119-R1 — tests
  > Severity: low
  > Evidence: `tests/unit/annotation_routing_test.py:488` iterates `FIXTURES.parent.rglob("*.py")`. `FIXTURES` is `tests/fixtures`, so this scans `tests/fixtures/**/*.py`, not the required `tests/**/*.py`.
  > Finding: The optional-provider import guard does not scan the test suite as required by T-119 acceptance criteria.
  > Failure/impact: A future test could import `cltk`, `stanza`, `torch`, `openai`, `anthropic`, or `ollama` without this guard detecting it.
  > Recommended direction: Scan from the `tests` directory and retain the synthetic-source negative control.

  A fresh candidate-validation `code-reviewer` Task rejected T-119-R1:
  `annotation_apply_test.py:18` sets FIXTURES to tests/fixtures, so its parent
  is tests. Both parent and validator executed the scan: 80 Python files,
  including the routing unit test; negative controls detect forbidden imports.
  Accepted findings remain none; disposition is rejected/not reproducible, not
  deferred. No code change or finding-fix TDD is appropriate for this false
  positive. A final fresh verifier checks this rejection before finalization;
  the review/fix/recheck limit remains at most two cycles.
- Second/final fresh disposition-verification Task independently confirmed the
  rejection, loaded its `code-reviewer` skill, Python reviewer/patterns, common
  review rules and TDD references, and ran the import guard plus synchronized
  writer test: 2 passed. Final conclusion verbatim: "No concrete task-relevant
  findings." Accepted/deferred findings: zero. Two verification passes, no
  finding-fix cycle required and no unresolved issues. At 2026-10-16T12:29:28Z,
  state validation, active selector, spec hash and fetched base were rechecked
  before lifecycle finalization; all still matched this task's ownership.
- 2026-10-16T12:29:50Z: verification pass
- 2026-10-16T12:29:59Z: Verified pass at 2026-10-16T12:29:50Z after final exact ruff/mypy/pytest2418 and mise check, independent General/Security/persistence-concurrency review, candidate validation and fresh final disposition verification; no validated or deferred findings. Full safety drift and invalid-pending rejection under existing locks; matrix and concurrency evidence recorded.
