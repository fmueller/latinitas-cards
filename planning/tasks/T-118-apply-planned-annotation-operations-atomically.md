---
id: T-118-apply-planned-annotation-operations-atomically
title: Apply planned annotation operations atomically with idempotent reruns and busy locks
status: completed
priority: high
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies:
    - T-117-plan-annotation-patches-against-policy-and-approve
    - T-115-record-corpus-claim-decisions-through-claim
updated_at: "2026-10-16T11:54:42Z"
---

# T-118-apply-planned-annotation-operations-atomically Apply planned annotation operations atomically with idempotent reruns and busy locks

## Description

Implement `annotations apply PLAN --corpus CORPUS --sidecar SIDECAR [--policy FILE]
[--receipt RECEIPT] [--timestamp T] [--json]` from
`specs/v0.3.0.md#agent-ready-cli-contract` and the writer rules of
`#sidecar-annotation-model`: recheck bindings, accept the base sidecar or one advanced
only by this plan, append `auto` and receipted operations atomically, and make reruns
`already_applied`. It implements Fixed Contracts **Exit codes** (`binding_drift`, `busy`,
`denied`, precedence 3 > 4 > 5 > 1) and **Runtime rules** (`created`). It consumes plans
and receipts (T-117), the writer (T-113, `atomic_io`) and must leave claim decisions
untouched (T-115). This task owns apply's locked snapshot/write transaction. T-119
implements pending-operation policy/safety-fact revalidation inside that transaction,
plus the cross-command actor/invalid matrix; it is not merely a test-only follow-up.

## Acceptance

- Rechecks, in precedence order: an unreadable or tampered plan (`invalid_plan`) or
  receipt (`invalid_receipt`), a plan containing any `invalid` operation
  (`invalid_operation`), a receipt whose `plan_id` differs (`invalid_receipt`), or a
  corpus with findings (one `invalid_corpus` error whose message names the first finding
  code, plus the findings) exits 3; the embedded patch is checked against `base.patch` as
  part of plan integrity (mismatch: exit 3 `invalid_plan`); a corpus header digest or
  policy digest (`--policy` given versus plan `null`, or a different file) differing from
  the plan's `base` exits 4 `binding_drift`; a receipt actor refused by the plan's
  `receipt_actor` exits 5 `actor_refused`. A receipt listing a `deny` operation exits 5
  `denied`; one listing an unknown ID exits 3 `unknown_operation`, and an `auto` ID is a
  no-op as at approve. Every such exit leaves SIDECAR byte-identical.
- Sidecar acceptance: SIDECAR is accepted when its digest equals `base.sidecar` (absent
  file when `null`), or when a line-boundary prefix has that digest and every record after
  it carries `plan` equal to this `plan_id`. A foreign change (an `annotations segment`
  run adding a passage, or one hand-appended record from another plan) and a corpus edit
  after planning both exit 4 `binding_drift` with no byte change. Apply also rechecks each
  record it would append against the current corpus: a missing passage or a `passage_hash`
  different from the record's `input_hash` exits 4 `binding_drift`. The corpus-change test
  re-normalizes an edited source (header `source.sha256` changes) and separately edits a
  passage text with a recomputed `passage_hash`/`norm` (header unchanged); both exit 4 with
  no byte change.
- Appending: `auto` operations and `review` operations listed in the receipt are written
  in plan order with `plan` = `plan_id`; `created` comes from the patch, else
  `--timestamp`, else `SOURCE_DATE_EPOCH`, else exit 2 `timestamp_required` (only when a
  record needs it); the timestamp grammar and `SOURCE_DATE_EPOCH` rule of T-113 apply.
  Records whose `annotation_id` already exists are skipped. `deny` operations produce
  `denied` warnings; unreceipted `review` operations are listed as pending; exit 0 in all
  these cases.
- Flow test on `tests/fixtures/annotations/mixed-valid.patch.json` (`mixed.patch.json`
  without its invalid operation) against `mixed.base.annotations.jsonl`, under
  `tests/fixtures/policy/mixed.policy.json` (both from T-117): plan; apply without receipt (auto
  records appended, review pending); approve the review operation as `human:reviewer`;
  apply the same plan with `--receipt` (accepted because the appended records name the
  plan; review record appended); apply again: `result.already_applied: true`, exit 0,
  sidecar byte-identical. Negative control: a plan from a changed patch does append.
  Replanning the identical patch against the advanced sidecar and applying adds no record.
- Hold `<CORPUS>.lock`, `<SIDECAR>.lock` and `<CORPUS>.claims.json.lock` exclusively,
  nonblocking, in lexical resolved-path order, before reading current corpus, sidecar
  or decisions through revalidation and replacement. Contention on any one exits 4
  `busy`, with no data changes; release earlier locks on later-lock failure. Never unlink
  locks. Corpus normalization shares the corpus lock; `claims record` keeps its blocking
  lock on the same claims path. Test each held lock and a competing normalizer/claim
  writer; no snapshot change can interleave before sidecar replacement. A failure
  injected into `os.replace` keeps previous sidecar bytes and leaves no temp file.
- Determinism: plan and apply run in two subprocesses with different working directories,
  `TZ`, `LC_ALL`, `PYTHONHASHSEED` and relative versus absolute paths, with one
  `--timestamp`, give byte-identical sidecars and plans free of absolute paths, hostnames and
  wall-clock values.
- Scope: a directory snapshot test shows apply changes only SIDECAR and may create the
  corpus, sidecar and claims lock files above, including when claims are absent;
  CORPUS, `<CORPUS>.claims.json`, CSV, managed state and package data are never opened
  for writing, and no claim decision is recorded. Protect all inputs, manifests and locks
  against direct/symlink/hard-link output aliases. Matrix row `annotations apply: 0,
  2, 3, 4, 5`.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record sidecar SHA-256 values after each flow step and both determinism runs.

## Implementation Notes

- New `annotation_apply.py` (binding checks, prefix scan, append selection) and an `apply`
  callback in `commands/annotations.py`; reuse `annotation_sidecar.append_records`,
  `resolve_created`, `annotation_plans` loaders and `policy_digest`.
- Use T-103's multi-lock primitive and a locked append/replace path; allow a narrow
  `annotation_sidecar` integration edit to avoid reacquiring an already-held lock.
  Leave a shared evaluation seam for T-119 to compare current pending-operation validity,
  route and complete persisted `safety` facts before any append, under these same locks.
  Do not introduce a claims digest or write decisions.
- Report `already_applied` when no record would be appended and every selected operation's
  records are already present; report `pending` operation IDs and `denied` warnings in
  `result` and diagnostics.
- A receipt plan mismatch is treated as exit 3 (rejected input), not drift; document it in
  the matrix notes. Size guard: one module plus one callback and necessary existing
  sidecar/shared I/O integration for the multi-lock transaction. Lock correctness cannot
  be deferred by the former single-module limit.

## Execution evidence (2026-10-16)

- Started at 11:19:16Z after `mise exec -- taskrail validate` and `next --json`
  selected exactly T-118, with no conflicting active owner. HEAD and fetched
  origin/main were `b35d523fd4658b00222eaa641e28379edcfe97ee` before start and
  finalization. Spec SHA-256 remained
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
- Loaded workflow-v3 and autonomous-backlog. The initial IO Delegate factual
  extraction was not dispatched (`subscription_approval`); no provider changes
  or equivalent bulk extraction were made. Resumed the same cycle using narrow
  implementation/review reads, not whole-source reconstruction: loader and plan
  identity, record derivation/evaluation branches, append locking, timestamps,
  POSIX lock/replace primitives, callback registration, and claims lock ownership.
  Spec reads covered the task's sidecar/CLI contracts and relevant fixed
  identity, files and runtime rules, not the full spec.
- Strict red/green: initial mixed flow failed because apply was unregistered;
  implemented module/callback and transaction integration. Record-integrity versus
  drift regression failed with exit 4, then passed with invalid_plan/3. Reserved
  lock/manifest output tests failed with successful writes, then rejected/3.
  Mixed replan initially failed target_inactive; preserved the existing unplanned
  inactive-target rejection while permitting already-plan-applied records.
- Lock-removal in-memory mutation: all three held-lock tests plus competing
  normalizer test failed (4 failures). Disabling the prefix guard caused foreign,
  non-boundary and segment suffix regressions (3 failures). Restored production
  code passed. These mutations changed no repository file. Tests cover blocked
  real claims writer, persistent locks/release on contention, actual edited-source
  renormalization, header-preserving passage edits/missing passages, replacement
  failure cleanup, aliases, absent sidecar, timestamp/dedup and directory scope.
- Dedicated code-simplifier Task loaded its skill and replaced repeated scans
  with ID lookups; its 144 targeted tests, ruff and mypy passed. Integration caught
  its exclusion of Retraction IDs: added a failing retraction apply/replan test,
  restored all planned record IDs, and passed the test. No broad refactor accepted.
- Separate parallel read-only code-reviewer Tasks loaded their skill and mapped
  General, Security and Database/filesystem persistence/concurrency guidance.
  General included Python companions; a separate Python lane was redundant.
  SQL/framework/ML lanes had no changed implementation. Fresh candidate validator
  validated SEC-1/SEC-2 and rejected DB-1; no candidate silently omitted.

### Verbatim validated findings and dispositions

- **T118-SEC-1 (fixed):** “A modified plan can change an auto operation’s annotation,
  recompute its `annotation_id` and `plan_id`, and still pass apply’s integrity checks,
  allowing a record unrelated to the embedded patch to be appended automatically.”
  Evidence: original plan identity lines 124–128 and apply lines 130–141.
  Rehashed value/producer/missing-record regressions all initially succeeded
  incorrectly (3 failures), then rejected invalid_plan/3 with byte-identical sidecar.
  Expected add/supersede records now derive from the embedded patch; retractions
  check patch-controlled fields and the retained target inside the locked snapshot.
- **T118-SEC-2 (fixed):** “When both conditions are present, apply reports
  `invalid_receipt` rather than the earlier-priority `invalid_operation` specified
  in T-118’s recheck order.” Evidence: original apply lines 72–75. Combined-condition
  regression initially returned invalid_receipt, then passed with invalid_operation
  and no write. Receipt structure still validates before either semantic check.
- **DB-1 (rejected candidate):** “Applying a plan to a valid sidecar whose final
  record has no trailing newline joins the first new record to that record,
  corrupting the sidecar.” Fresh validation found read_sidecar lines 250–253
  already reject noncanonical or unterminated lines before writes. No separator
  insertion or prefix-boundary weakening was made.
- General: “No concrete task-relevant findings.” Fresh disposition verifier
  loaded code-reviewer and General/Python/Security guidance, verified both fixes
  and DB-1 rejection, ran 111 planner/apply tests, and reported “New findings: None.”
  First-cycle verification; no deferred validated findings at that point.

### First-cycle gates and real CLI flow

- First-cycle `uv run ruff check` → `uv run mypy` → `uv run pytest -v` passed:
  2,377 tests, one existing Pydantic serialization warning. `mise run check`
  independently passed all gates: 373 files formatted, 161 mypy source files,
  2,377 tests, commit/push/author and mutation policy guard suites. The first full
  run's registration/matrix failures were fixed by adding the apply callback
  registration and executable 0/2/3/4/5 matrix; no legacy expectations weakened.
- Subprocess mixed flow at 11:38:25Z, timestamp 2026-01-02T03:04:05Z:
  auto appended 1 with one review pending; human:reviewer receipt appended 1;
  rerun appended 0/already_applied; replan appended nothing. Sidecar SHA-256:
  - Base: `cb822ae209d8905721a582b3e271582f8da76f9226a1c017ccf167afb617ce4e`
  - Auto: `32b29ffda2d2a271cbc9de3ea9a50f79b550e712c770b88219bffbee775a8283`
  - Reviewed/rerun/replan: `fcf222b066850ff4e5c3ed1556f2bbb4093ea2366fc01bafd3bded9bea77a252`
- Both subprocess determinism runs (UTC/C/seed12/relative and
  Europe/Zagreb/C.UTF-8/seed98/absolute, different working directories) matched:
  plan SHA-256 `8e115f3c0c71b9f0b3129c4bed952e47b8733366b134bd903391ec3b344dd3a0`;
  sidecar SHA-256 `32b29ffda2d2a271cbc9de3ea9a50f79b550e712c770b88219bffbee775a8283`.
- Scope: sidecar writes/locks only; no claim decisions, CSV, managed/package writes.
  T-119 retains ownership of pending route/complete safety revalidation inside
  this transaction. T-127 exporter, T-128 future memory work and T-038 release
  gate were not changed. Hooks installed under the maintainer identity.
- 2026-10-16T11:44:08Z: verification pass
- 2026-10-16T11:44:17Z: Verified 2026-10-16T11:44:08Z after workflow-v3 simplification, independent lanes, fresh candidate/disposition verification and all mandatory gates; T119 remains the safety-revalidation follow-up.

### Second-cycle exact claims-lock correction

- A final integration source check found the legacy claims writer derives its
  lock from the logical claims path, whereas writer_locks normally resolves the
  data symlink first. The initial pre-push was already running and completed;
  the correction is a second focused commit, not a rewrite of published history.
- Regression `test_exact_claims_lock_when_decisions_are_symlinked` held the real
  legacy flock and initially failed: apply returned 0 instead of busy/4. The
  correction lets the same shared primitive take explicit lock paths; apply
  supplies resolved corpus/sidecar locks and the exact logical claims lock.
  Default callers are unchanged. The regression now passes without sidecar
  changes or a redundant lock beside the symlinked data target.
- Separate second-cycle General, Security and filesystem persistence/concurrency
  code-reviewer Tasks loaded their mapped guidance and each reported
  “No concrete task-relevant findings.” Fresh candidate validation independently
  validated the discovered defect and its correction, with this explanation:
  “With the claims data path symlinked, deriving its lock from the resolved target
  would miss the logical `<corpus>.claims.json.lock` used by `record_corpus_batch`.”
  Disposition: fixed with the failing/passing regression above. All 93 focused
  apply/atomic-I/O/claims tests, ruff and mypy passed. This is cycle two of two.
- Fresh second-cycle disposition verifier loaded General/Python/Security and
  Database/filesystem guidance and companions, independently ran 93 focused tests,
  confirmed the exact-lock correction, both prior SEC fixes and DB-1 rejection,
  and reported “No concrete task-relevant findings.” No unresolved findings remain.
- Final full `uv run ruff check` → `uv run mypy` → `uv run pytest -v` passed
  2,378 tests (one existing warning). Final `mise run check` independently passed
  2,378 tests, formatting, types, and every commit/push/author/mutation guard suite.
  Spec hash stayed pinned; fetched origin/main matched the initial delivered
  T-118 commit before the focused correction. Next selector remains T-119;
  no second task was started and no follow-up task was created.
- 2026-10-16T11:54:42Z: verification pass
- T-118 remains completed from 11:44:17Z and was reverified at 11:54:42Z after
  cycle two. A repeated complete request was refused as non-transitionable
  because the task was already completed; no status fields were hand-edited.
