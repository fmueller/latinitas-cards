---
id: T-204-check-v0-5-0-release-readiness-without-publication
title: Check v0.5.0 release readiness without publication
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-203-audit-v0-5-0-semantic-and-acceptance-gaps
updated_at: "2026-11-02T23:10:11Z"
---

# T-204-check-v0-5-0-release-readiness-without-publication Check v0.5.0 release readiness without publication

## Description

P50-7 readiness owner consolidates evaluated, integrated and audited evidence. Prepare
a candidate only, with no push/tag/release/deployment/provider authority. Follow
docs/releasing.md and spec execution-isolation/final-check requirements.

## Acceptance

- Confirm closed drift/gaps and exact candidate evidence for P50-1 through P50-6,
  held-out criteria, real library/authorized endpoints, native Desktop, compatibility
  and approved limitations. Missing prerequisites block readiness.
- Verify package build/install and CLI smoke from disposable built artifacts, docs/
  examples/schema registry, runtime prerequisites, migration/manual resolution and
  rollback instructions. Record candidate/artifact hashes.
- Run mandatory ruff/mypy/pytest chain and applicable build/policy checks. Semantic
  fixes require affected evaluation/integration AND both audits refreshed; stale
  evidence cannot attach to a changed candidate.
- Prepare concise user-facing changelog/version/release docs under the runbook.
  Distinguish preparation from publication; no unsupported Mobile/scheduled-collection/
  zero-error claims.
- Request exact-candidate maintainer readiness acceptance and list remaining credentials,
  CI/environment permissions and recovery steps. Publication stays separately blocked.

## Verification Notes

Check runbook/audit matrix, hashes and install smoke on the same candidate. Report actual
pass/fail/blocked results. Readiness cannot replace linguistic/native evidence or grant
permission to publish.

## Implementation Notes
