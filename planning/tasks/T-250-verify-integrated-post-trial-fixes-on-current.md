---
id: T-250-verify-integrated-post-trial-fixes-on-current
title: Verify integrated post-trial fixes on current transport
status: blocked
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-249-evaluate-bounded-pydantic-ai-migration-feasibility
updated_at: "2026-11-28T00:36:19Z"
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
