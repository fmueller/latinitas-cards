---
id: T-202-audit-v0-5-0-contract-and-implementation-drift
title: Audit v0.5.0 contract and implementation drift
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-201-validate-real-pipeline-and-linux-anki-desktop
updated_at: "2026-11-02T23:10:11Z"
---

# T-202-audit-v0-5-0-contract-and-implementation-drift Audit v0.5.0 contract and implementation drift

## Description

Final drift checkpoint after features, held-out and real/native acceptance. Owner
compares code, schemas, CLI registry, docs and evidence to exact accepted spec/design
revisions. Structural coverage is a signal, not semantic acceptance.

## Acceptance

- Audit binding decision/reuse/provider/selection/ownership/curated/Anki contracts and
  exclusions. Inspect strict-reader upgrades, historical human decisions, identities,
  v0.4.0 German goldens and unchanged offline/deck workflows.
- Compare actual limits, hosted permission/data scope, no-network behavior, full-assertion
  and curated gates against agreed contracts, including negative controls and handoffs.
- Record drift with source/evidence revisions, impact and resolution. Explicitly reconcile
  approved design deltas. Unresolved release-impacting drift cannot pass.
- Fix prerequisites of remaining release work there or through tracked spec-anchored
  dependencies, not post-release follow-ups. Re-run affected scenarios and the full
  mandatory chain after code fixes.

## Verification Notes

Run taskrail validate/coverage and inspect active spec; inspect semantic contracts/evidence
separately. Later contract changes require audit refresh before readiness/publication.
Record run timestamps, not committed gitignored artifact paths. Execution-isolation and
final-check rules remain binding.

## Implementation Notes
