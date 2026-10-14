---
id: T-118-apply-planned-annotation-operations-atomically
title: Apply planned annotation operations atomically with idempotent reruns and busy locks
status: todo
priority: high
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies:
    - T-117-plan-annotation-patches-against-policy-and-approve
    - T-115-record-corpus-claim-decisions-through-claim
updated_at: "2026-10-14T05:42:59Z"
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
