---
id: T-217-audit-deck-follow-up-compatibility-and-release
title: Audit deck follow-up compatibility and release readiness
status: todo
priority: medium
spec_ref: specs/v0.5.1.md#delivery-and-acceptance-plan
dependencies:
    - T-216-accept-deck-follow-up-workflows-with-linguistic
    - T-204-check-v0-5-0-release-readiness-without-publication
updated_at: "2026-11-05T22:11:24Z"
---

# T-217-audit-deck-follow-up-compatibility-and-release Audit deck follow-up compatibility and release readiness

## Description

Independently audit drift, coverage, docs and readiness after whole-spec acceptance.
This task grants no publication authority.

## Acceptance

- Compare public contracts to approved revisions and all requests; maintain
  bidirectional clause → task → scenario → result/evidence mapping. Resolve semantic
  gaps instead of treating structural coverage as acceptance.
- Verify unchanged v0.4.0 profiles/German goldens, stable identities/verb ordinals,
  fail-closed drift, migration/recovery and native scheduling evidence. Reject stale/
  absent evidence and rerun affected checks after fixes.
- Supply runnable docs for all requested workflows and limitations, accurate
  checkpoint semantics and documented unsupported input-specific outcomes.
- Require v0.5.0 readiness, exact-candidate linguistic/native acceptance, findings
  closed and maintainer readiness acceptance. Build/install/check artifacts and record
  hashes and mandatory-chain results on the same candidate.
- Do not publish/tag/push/trigger releases or unblock maintainer gates. Publication
  requires a separate authorized release task/procedure after readiness.

## Verification Notes

Read-only drift/gap audit followed by approved local fixes and affected reruns.
Unavailable native/reference evidence blocks readiness; report failed/not-evaluable.

## Implementation Notes
