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
