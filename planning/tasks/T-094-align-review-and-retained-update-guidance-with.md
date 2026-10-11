---
id: T-094-align-review-and-retained-update-guidance-with
title: Align review and retained-update guidance with supported CLI
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies: []
updated_at: "2026-10-11T19:29:13Z"
---

# T-094-align-review-and-retained-update-guidance-with Align review and retained-update guidance with supported CLI

## Description

ROUND2 found three related stale capability statements while following the public
workflow on 45a7cbfb7a73db3aab77ab838f94423829dfdfe6. Severity: low; confusing user
guidance, not a failed retained update. Keep this a focused documentation/CLI
wording correction, not a capability expansion or general documentation review.

- `docs/contextual-form-parsing.md` says there is currently no CLI command to
  author decisions and instructs manual JSON editing. Shared `claims record`
  successfully authored and corrected the fresh parsing fixture.
- `docs/reference-note-type.md` says the current journal rejects changed per-card
  answers as unsupported slot effects, even after CSS setup. Fresh approved
  retained principal/parsing/phrase updates completed emit/observe/reconcile.
- Public principal `preview --profile PROFILE --input SOURCE` prints
  `retained-slot payload application unsupported`. The approved retained path
  is supported; structural creation/retirement and layout migration remain blocked.

Expected: accurately distinguish supported claim recording and approved retained
payload updates from initial manual setup, structural changes and native safety.
Dedup: T080/T081/T089 completed guidance work does not track these remaining
contradictions. T090/T091 fixes do not own these statements.

## Acceptance

- Point parsing authors to the supported shared claim CLI without removing the
  legitimate explicit manually authored JSON route or individual-review requirement.
- Reference setup and principal preview describe the separately approved retained
  update path accurately while retaining strict exact layout/CSS binding and manual
  structural guidance. Do not imply ordinary overwrite CSV safely updates a deck.
- Preserve explicit native acceptance/sync/restore limits; no new transport,
  migration or native verification claim.
- Check changed CLI wording through a fresh supported preview and cross-check the
  named commands against current help and the exercised retained workflow.

## Verification Notes

- Fresh ROUND2 full protocols observed 44 principal/supine/parsing/defective and
  24 phrase operations with zero unresolved operations; closed-storage fixture
  evidence does not prove native CSV import or scheduling/sync/restore safety.
- Captured principal, supine and defective preview logs all contain the stale
  warning. Exact excerpts, commands and outputs are retained in the ROUND2 bundle.
- Native Desktop/Mobile remains maintainer-deferred/pending. No product fixes or
  task verification performed by tester.

## Implementation Notes
