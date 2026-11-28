---
id: T-250-verify-integrated-post-trial-fixes-on-current
title: Verify integrated post-trial fixes on current transport
status: blocked
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-249-evaluate-bounded-pydantic-ai-migration-feasibility
updated_at: "2026-11-28T02:01:46Z"
---

# T-250-verify-integrated-post-trial-fixes-on-current Verify integrated post-trial fixes on current transport

## Description

A1 independent offline acceptance of post-trial fixes on the frozen installed
current-transport candidate. Requires accepted F1/F2 and recorded M1 outcome, not
M2 or a hosted replay. This owner must differ from F1/F2 implementers. See
docs/v0.5.0-post-trial-plan.md.

## Acceptance

- Derive expectations from the approved spec, then exercise installed public pipeline
  boundaries with synthetic German production/recognition, multi-binding phrases,
  English compatibility, nested/malformed/presentation failures and unsupported Latin
  withheld. Inspect outputs and persisted state, not only unit mocks or green CI.
- Exercise synthetic historical-layout read/resume, strict corruption rejection,
  exact permissions, retained failed/unknown charges, one-card blind independence,
  unchanged-scope reuse and offline no-network behavior. Use representative happy
  paths/important negatives rather than another broad matrix. Never commit private
  collection/store/model-response fixtures.
- Capture exact source/wheel/installed identity, commands, expected/observed results,
  coverage and evidence gaps. Prior task-local checks are not independent acceptance.
  Return confirmed defects to their owner under implementing-changes; rerun affected
  scenarios on the corrected frozen candidate.
- Complete on passed offline acceptance alone, including M1 defer/infeasible outcomes.
  Do not wait for R1 or claim hosted quality, linguistic authority, protected T200
  evaluation, native Anki acceptance or release readiness.
- T201 remains blocked pending accepted A1 and M2 if M1 adopts. R1 stays separately
  permissioned; declined/unperformed replay is not passed. Adopted M2 must independently
  repeat these offline controls on the migrated candidate and satisfy separate
  qualification obligations.

## Verification Notes

Follow implementing-changes for any product fixes and use installed acceptance for
this verification task. Missing material evidence blocks acceptance. Final Ruff →
mypy → full pytest precedes verify/complete. No hosted inference, collection edits,
import or publication. Commit/push evidence/task completion on main without waiting
for CI; parent owns final CI-fix supervision.

## Implementation Notes

- Candidate: origin/main 32f1af090dd754a97c81d36bbce833ee45f9e033. Built wheel
  sha256 2ae70f43a2adee5403fe9fcc52bf9e71841f3fd66655e70b079417b56bacfb40
  (sdist 979aa18602ce7d3bc77f1f64ef7fdda66b8e17b4ea384b4a8a07852594d0a02f),
  installed into a separate Python 3.13.5 venv with locked runtime constraints.
  All CLI runs used a loopback-only network namespace, a scrubbed environment
  without API keys or proxies, a scratch HOME and a synthetic local endpoint
  with hand-derived answers. No hosted call, collection, import or replay.
- Passed on the installed candidate: German and English production and
  recognition with explicit direction and only the applicable template;
  byte-identical proposal-free blind tasks; two- and three-binding phrases
  with distinct entries; malformed, duplicate-key and partial-sense output as
  `malformed_output`; direction, stem-line and concatenation breaks as
  `contract_violation`; consistent wrong Latin parsed but withheld; 503 and a
  stalled body under a 2 s deadline retained as charged unknowns with timing
  marks and no redispatch on resume; unchanged-scope resume with identical
  calls and charges; a store written by the 72fdf19 wheel read as schema 2,
  resumed offline with no dispatch and online with only the changed
  proposal/2 call while old calls and charges stayed byte-identical; stage,
  response, ledger and duplicate-key corruption rejected without dispatch;
  owner-only run, provider and campaign directories (0700, files 0600), with
  widened directories refused; LLM-off and `--offline` made no requests.
- Blocking defect (pre-dates the post-trial package; also present at 72fdf19):
  a controlled phrase whose bindings reuse one source entry (verb-noun
  subject and object, or noun-genitive head and dependent, from the same
  noun) aborts the provider path before dispatch. `pipeline run` exits 1 with
  `duplicate source_evidence` from ModelTask validation, writes no outcome
  report and records no calls or charges; the same inputs complete offline.
  `_task` adds identical per-binding `unverified-phrase-entry` evidence for a
  repeated entry. Committed provider-path phrase tests bind distinct entries.
  Returned to the parent for owner assignment; affected phrase scenarios must
  be rerun on a corrected candidate.
- Not claimed: hosted quality, Latin authority, T200 evaluation, native Anki,
  release readiness, R1 or M2. Verify/complete not run; the final chain was
  stopped after Ruff and mypy passed and pytest had 334 passing, 0 failing.
- 2026-11-28T00:36:19Z: A1 defect: phrases whose bindings reuse one source entry abort the provider path (duplicate source_evidence in ModelTask, pre-dates package); needs owner correction and rerun of affected phrase scenarios on the corrected candidate
- 2026-11-28T01:52:21Z: T-252 correction on origin/main 6aca005; resuming A1 on the corrected candidate
- Round 2 candidate: origin/main 6aca005c028430c4f99d3807473eac052d20e3f0
  (T-252). Wheel sha256
  a423025cc4f3f92fd8452ddf64c59b276f9a44fdea383764cc9c357983c2c885, sdist
  b0e41153327c7b5a071bee2a46bbbe22ea225c68178bc9607252403badc7d7e3, fresh
  venv. The installed package differs from round 1 only in
  pipeline_orchestration.py, and runtime pins are identical. Every round-1
  scenario was rerun on this install; round-1 results are not reused.
- Round 2 passed: same-entry verb-noun and noun-genitive phrases now send one
  proposal and one byte-identical blind check, keep each role's form cell and
  deduplicate only identical entry evidence; same entry with different senses
  keeps both meaning spans; a same-entry partial sense mapping is still
  `malformed_output`. Form, recognition and distinct-entry phrase requests
  equal round 1 except implementation-bound digests. Direction, nested-output,
  wrong-Latin, unknown/deadline accounting, unchanged-scope reuse, 72fdf19
  store read and resume, response/stage/ledger/duplicate-key corruption,
  owner-only permissions and offline controls behave as in round 1. A
  tampered current-store response fails only its card, with no dispatch.
- Round 2 blocking defect (same class as T-252): a phrase reference repeating
  an identical `constructed-exercise-context` aborts the provider run with
  `duplicate source_evidence` before dispatch, because contexts are appended
  after T-252's deduplication. The whole run stops, including an unrelated
  clean phrase card, and writes no outcome report; offline both cards
  complete as withheld. Two distinct contexts work. Returned to the parent.
- Harness note: copied runs still read and write report.json in the original
  output directory recorded in execution.json, so `report` results on copied
  stores were discarded; resume results are copy-local.
- Round 2 checks: no tests added or removed. Verify/complete not run because
  the task is blocked; the full Ruff, mypy and pytest chain was not run on
  6aca005 in this task.
- 2026-11-28T02:01:46Z: A1 round 2 on 6aca005: identical repeated phrase constructed-exercise-context still aborts the provider run (duplicate source_evidence; contexts appended after T-252 dedup), stopping unrelated cards; needs owner correction and rerun
