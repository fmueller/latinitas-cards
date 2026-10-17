---
id: T-138-reconcile-dependabot-maintenance-batch
title: Reconcile Dependabot maintenance batch and packaging inventories
status: in_progress
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-17T10:03:55Z"
---

# T-138-reconcile-dependabot-maintenance-batch Reconcile Dependabot maintenance batch and packaging inventories

## Description

Reconcile the five maintainer-requested Dependabot PRs against the packaging,
installation, portability, license, and release safeguards in
`specs/v0.3.0.md#pypi-publishing`. This is new maintenance, not a rewrite of
T-121's historical inventory or authorization to publish a release.

## Acceptance

- Handle #65, #72, #71, #70, and #63 serially, reconciling each exact head and
  current accepted main; require hosted PR and resulting main CI to pass.
- Preserve Python 3.13/3.14, the six default dependencies, optional NLP extras,
  meaningful lock/license inventory checks, and historical release evidence.
- Upgrade artifact consumers before the producer; retain mutation's exact
  upload-artifact v7.0.2 pin and all publication safeguards. Never trigger release.
- Record per-PR rebases, fixes, reviews, test evidence, and delivery checkpoints.

## Verification Notes

- Starting accepted base: 57efea7dd332ecf62e1f2efdf484d838f221a367.
  Hosted Build 39022901568 and CodeQL 39022901493 both passed on that exact head.
- Spec SHA256 remains fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704.

## Implementation Notes

- T-138 was the deterministic selector, with no other active task, before start.
- Ownership is limited to this batch; T-038 and T-128 remain excluded.
- Workflow steps 1-3: inspected actual diffs, dependency metadata, artifact
  compatibility, repository guards, and existing inventory acceptance tests.

### PR #65 preparation

- Rebased the existing urllib3 security branch cleanly onto accepted main
  57efea7dd332ecf62e1f2efdf484d838f221a367. No application source changes.
- RED: packaging/release inventory tests failed on the lock fingerprint and
  urllib3 2.7.0 versus 2.8.0 audit row. Strengthened historical snapshot/link
  acceptance failed before the current audit linked the preserved snapshot.
- GREEN: focused packaging/release acceptance: 6 passed. Exact urllib3 2.8.0
  wheel PEP 658 metadata declares MIT, LICENSE.txt, and Python >=3.10.
- Preserved original audit byte-for-byte in the dated snapshot; its SHA256 is
  pinned. Historical release documents and prior task evidence remain unchanged.
- Step 4: dedicated code-simplifier reported no safe meaningful simplification;
  no edits. Steps 5-6: independent General, Python, Security lanes each returned
  verbatim "No concrete task-relevant findings." Fresh candidate validation
  confirmed zero candidates; no fix or deferral dispositions were needed.
  Python covers test changes; Security covers dependency supply-chain behavior;
  database/framework/ML lanes omitted because none of those contracts change.
- Step 7: final ruff passed; mypy passed (169 source files); pytest passed
  (2625 passed, 7 skipped, 1 expected serializer warning, 286.40 seconds).
  Fresh disposition verifier returned "No concrete task-relevant findings."
- Sequential `mise run check` passed, including another 2625-pass suite
  (278.10 seconds), format, message/author, mutation, 54 distribution cases,
  and publication guards. `taskrail validate` reports state valid.
- Batch remains active; verify/complete wait for all PR dispositions and CI.

### PR #65 delivery / PR #72 preparation

- PR #65 head 84be9f46df32283f016bb8f0b13eb2abf5115794 passed all nine
  checks: Build 39024505482 and CodeQL 39024504561. Fresh reviews/comments
  had no blockers; exact-head rebase merge delivered
  48110d41b4b6453aad1f612a49289bd0214c084a, whose tree matches that head.
  Previous accepted main remains an ancestor. Resulting-main Build
  39025008366 and CodeQL 39025007978 passed before any PR #72 writes.
- PR #72 original head 822d966841a7549f818842957df4d76bc6c6046e rebased
  cleanly onto that accepted main. Scope: two release download v4 -> v8
  references; uploader remains v4, name/path remain dist/dist/. No source,
  dependency, test, trigger, permission, publication, or install-contract edits.
- Exact upstream action metadata confirms v8 uses Node24, default digest
  mismatch error, and default decompression enabled. The v4 uploader produces
  archived artifacts with digests; hosted ubuntu-latest supports the runner.
  Consumer-first compatibility is based on upstream API/source review, not an
  executed release workflow; no publication workflow is triggered for testing.
- Action-only validation: release_workflow_test.py 2 passed; actual publication
  guard and its adversarial synthetic suite passed. Existing acceptance tests
  are retained unchanged; no artificial version-string regression test added.
- Dedicated simplifier made no edits. General and Security review each returned
  "No concrete task-relevant findings." Candidate validation found no lost
  candidates; no dispositions. Fresh disposition verifier returned the same
  no-findings conclusion after focused tests/guards passed again. Language,
  database, and ML lanes omitted: only action references and task notes change.
- Exact-head hosted Build/CodeQL and resulting-main CI remain delivery gates.

### PR #72 delivery / PR #71 preparation

- PR #72 head cb0b524d08af4013727287e3b4e6ad938a567fab passed all nine
  checks: Build 39026068284, CodeQL 39026066584. Fresh head/base/reviews
  were unchanged and clean. Rebase merge delivered
  a2cae54233a4f2d93cd7b060615dfe62cfcd5aff; accepted base is an ancestor
  and tree exactly matches the validated head. Resulting-main Build
  39026560552 and CodeQL 39026560719 passed before PR #71 writes.
- PR #71 original head 245bc82a40ba69a12a8b183a0c11d548503dd9a0 rebased
  cleanly onto accepted main a2cae54233a4f2d93cd7b060615dfe62cfcd5aff.
- Rejected incidental mutation workflow v7.0.2 -> floating v7 change. A
  temporary byte-exact-to-main assertion failed before correction, then passed
  after restoring the existing pin. No new permanent version-string test.
- Final workflow scope is release producer upload v4 -> v7, paired with the
  already accepted download v8 consumers. Exact upstream action metadata
  confirms archive defaults true and Node24. Directory upload/name/dist paths,
  hash validation, tag-only release, publish environment/OIDC remain unchanged.
- Action-only checks: release workflow 2 passed; actual/synthetic publication
  guards and mutation configuration checks passed. No release run was invoked.
- Dedicated simplifier made no edits. Independent General and Security each
  returned "No concrete task-relevant findings." Candidate validation found
  zero candidates; no dispositions. Fresh final verifier returned the same
  no-findings conclusion, noting the pin correction must be committed before
  hosted CI. Language/database/ML lanes omitted: no such contracts change.
- Repeated focused checks passed; `mise exec -- taskrail validate` reports
  state valid. Hosted exact-head and resulting-main CI still gate delivery.

### PR #71 delivery / PR #70 preparation

- PR #71 head 79515ac1ba728536b4774e64a0a5f867b8e0fb28 passed all nine
  checks: Build 39027779787, CodeQL 39027776660. Fresh reconciliation showed
  unchanged clean base/head, no comments or reviews. Rebase merge delivered
  1ba40e38bb0ceebded6d109783c0c4bf468aeb29; accepted base ancestor and tree
  equality verified, including mutation v7.0.2. Resulting-main Build
  39028216714 and CodeQL 39028216576 passed before PR #70 writes.
- PR #70 original head 8228d562f4e7d7e36ddbf769586040993d70ec17 rebased
  cleanly onto that accepted main, preserving accepted urllib3 2.8.0.
- Actual version changes: click 8.4.2 -> 8.5.0; pydantic 2.13.5 -> 2.14.1;
  core 2.46.5 -> 2.50.1; typer 0.27.2 -> 0.27.3; torch 2.14.0 -> 2.14.1
  (PyPI and CPU); mypy 2.3.1 -> 2.4.0; librt 0.15.0 -> 0.16.0; ruff
  0.16.8 -> 0.17.0; pandas-stubs 3.0.0.260204 -> 3.0.5.260914;
  typing-extensions 4.15.0 -> 4.16.0; typing-inspection 0.4.2 -> 0.4.4.
  Click upper bound becomes 8.6 and Ruff upper bound 0.18; no extra churn.
- RED: existing packaging and release audit tests failed on supported Click
  bounds and the 12 changed audit records. GREEN: after exact current audit,
  notices, and fingerprint maintenance, all 6 focused tests pass. Historical
  audit checksum and prior release evidence are unchanged; no checks removed.
- Fetched exact updated wheel PEP 658 metadata, including PyTorch CPU-index
  metadata, without installing NLP. Python ranges accept 3.13/3.14; license
  expressions/files retain the documented terms, including composite PyTorch
  and unresolved NVIDIA obligations. Default six packages and extra boundaries
  unchanged; 93 total / 92 third-party records remain.
- Dedicated simplifier made no edits. Independent General, Python, Security
  each returned "No concrete task-relevant findings." Candidate validation
  confirmed zero candidates; no fix/deferral dispositions. Fresh final verifier
  returned the same no-findings conclusion. Database/framework/ML lanes omitted:
  no corresponding implementation or contract changes, only dependency versions.
- Final exact chain passed: ruff; mypy (169 source files); pytest (2625 passed,
  7 skipped, 1 expected serializer warning, 273.95 seconds). Historical audit
  and active-spec checksums independently rechecked unchanged.
- Sequential `mise run check` passed, including a 271.94-second full suite,
  format/type/lint checks, all policy/mutation guards, 54 distribution cases,
  and publication synthetic guards. Taskrail state validates with T-138 active.

### PR #70 delivery / PR #63 preparation

- PR #70 head b2830ac2fe9de7c53cba9ed758412eda1b55c657 passed all nine
  checks: Build 39029742139 and CodeQL 39029740123. Fresh reconciliation
  preceded rebase merge to ca0020995a3f22b607a71381169968b5f6f573c1;
  ancestry and tree equality verified. Resulting-main Build 39030222702 and
  CodeQL 39030222426 passed before PR #63 writes.
- Dependabot automatically rebased PR #63 to head
  cc16efd28386f5259ce41d6ef1ca816a3e870e59 on that accepted main. No local
  rebase conflict. Only pandas 2.3.3 -> 3.0.6 and pytz removal change the lock;
  the pandas bound remains >=2.3.3 and now permits <4.0.0.
- RED: existing packaging and current-audit tests failed on stale bound,
  inventory fingerprint, counts, and pandas/pytz rows. Reconciled the current
  audit/notices and exact assertions: 92 total / 91 third-party records,
  90 normalized third-party names. Historical audit and spec hashes unchanged.
- Checked exact pandas 3.0.6 wheel metadata (Python >=3.11; cp313/cp314
  artifacts available); upstream v3.0.6 LICENSE confirms BSD-3-Clause while
  artifact license expression remains unstated. No NLP dependency added to
  defaults, no optional model installed, no runtime implementation changed.
- Strengthened the existing public split CSV test with asymmetric two-row
  expansion, NA/NULL/empty values and carried Unicode. Deliberately removing
  keep_default_na=False at runtime produced the expected failure: NA/empty
  became nan. Restored ordinary runtime passes. All 57 focused CLI,
  packaging, and release-candidate tests pass; uv lock --check also passes.
- Dedicated simplifier made no edits. Independent General, Python, Security
  each returned "No concrete task-relevant findings." Fresh candidate
  validation confirmed zero candidates and no dispositions. Database/framework/
  ML lanes omitted: no corresponding implementation or contract changes.
- Final exact chain passed: ruff; mypy (169 source files); pytest (2625 passed,
  7 skipped, one expected serializer warning, 273.06 seconds). Sequential
  `mise run check` passed, including full pytest in 271.99 seconds, formatting,
  type/lint checks and distribution/policy/publication synthetic guards.
- Fresh disposition verifier returned "No concrete task-relevant findings."
  No findings were deferred; hosted exact-head and resulting-main CI still
  gate delivery. Taskrail finalization waits for all five accepted deliveries.
