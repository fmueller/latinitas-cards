---
id: T-186-approve-v0-5-0-contracts-and-evaluation-policy
title: Approve v0.5.0 contracts and evaluation policy
status: blocked
priority: high
spec_ref: specs/v0.5.0.md#design-approval-and-execution-isolation
dependencies:
    - T-184-design-v0-5-0-acceptance-and-pipeline-contracts
    - T-185-design-independently-judged-acceptance-evaluation
updated_at: "2026-11-02T23:10:47Z"
---

# T-186-approve-v0-5-0-contracts-and-evaluation-policy Approve v0.5.0 contracts and evaluation policy

## Description

Maintainer-only design checkpoint. Only the maintainer may unblock after reviewing
the exact contract/protocol revisions. This is separate from planning approval and
spec activation. No autonomous loop may approve or unblock it.

## Acceptance

- Record explicit maintainer adoption of the sequencing amendment and exact P50-1
  engineering/method revisions: schemas/upgrades, comparison/check contracts, finite
  limits, selection/batching, ownership/migration, curated approvals and handoffs.
  Use docs/v0.5.0-decision-package.md and the forward evaluation plan; retain this
  task's T-185 dependency and require design verification against adopted criteria.
- Resolve engineering choices with no enabled automatic policies. Missing empirical
  qualification, permitted acquisition, custody/output feasibility and justified
  numerical criteria remain explicit T-200 checkpoints, not waived evidence or
  an approval of automatic export. Preserve historical failures and exact revisions.
- Require policy denial at decision/reuse/export/handoff, isolated synthetic positive
  tests and no shipped test-policy authority. Neither stored success nor a model,
  profile or stage command can approve a policy. Existing human/curated gates remain.
- Record future native/library/provider/release prerequisites without treating them
  as satisfied or granting hosted-processing/publication permission.
- Reconcile downstream bodies with the approved contract. Changed policy requires
  refreshed affected review, reuse invalidation and maintainer acceptance before
  dependent consumers proceed.

## Verification Notes

Inspect signed-off revisions and unresolved-decision lists, not status alone.
Every implementation consumer depends transitively on this gate. The maintainer
separately authorized v0.5.0 activation on 2026-11-03; that activation does not
approve these contracts or unblock this gate.

## Implementation Notes

- 2026-11-02T23:10:47Z: maintainer design approval: accept exact contract and independently judged evaluation revisions before consumers
