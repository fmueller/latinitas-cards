---
id: T-095-correct-remaining-retained-payload-transport
title: Correct remaining retained-payload transport limitation
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies: []
updated_at: "2026-10-11T20:57:08Z"
---

# T-095-correct-remaining-retained-payload-transport Correct remaining retained-payload transport limitation

## Description

ROUND3 confirmed a remaining public transport-contract contradiction on
8d33fc885926bea5cb5447493d0b8511d4c344e0. Severity: low (documentation),
not a functional retained-update failure. The spec requires a safe approved
retained-slot payload-update path and accurate manual structural boundaries.

Reproduce with `sed -n '352,368p' docs/destination-snapshots-and-file-transport.md`
and `sed -n '444,451p' docs/destination-snapshots-and-file-transport.md`.
Lines 357-360 correctly permit separately approved retained prompt/answer
changes. Lines 447-450 instead list all slot prompt/answer/guard edits as
unsupported. Expected: distinguish supported retained payload changes from
unsupported guard, eligibility, structural and layout changes.

Fresh synthetic public CLI capture/adopt/plan/approve/emit/observe/reconcile
completed 44 principal/parsing and 24 phrase operations, zero unresolved.
This contradicts the broad prohibition, without certifying native import.

Dedup: completed T094 corrected three explicitly named surfaces (parsing
guidance, reference note-type guidance, principal preview), not this transport
document. Those regressions pass. T048/T059 are historical original contracts;
T080/T081/T089 and T090-T093 do not track this remaining sentence. Do not
reopen or edit completed tasks or widen this into general documentation review.

## Acceptance

- Correct the remaining transport limitation to permit separately approved
  retained principal/parsing/phrase prompt/answer payload updates.
- Preserve unsupported guard/eligibility/slot creation, retirement,
  reactivation, schema migration and APKG-application boundaries.
- Preserve exact CSS/layout binding, manual setup and native matching/tag,
  sync/restore limits. Native appearance remains pending, not verified.
- Cross-check the two passages against the exercised retained workflow;
  no new transport capability, lifecycle change or broad documentation rewrite.

## Verification Notes

- ROUND3 pinned v0.2.1 spec SHA256:
  0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4.
- Five fresh closed-storage rehearsals preserve 47 complete card rows and
  47 history rows, source notes, Personal Notes and user tags. These are
  offline protocol checks, not native CSV importer/scheduling/sync proof.
- Local task proposal only; parent acceptance required before planning-only
  commit/push. Tester made no product fix or manual lifecycle/state edit.

## Implementation Notes
