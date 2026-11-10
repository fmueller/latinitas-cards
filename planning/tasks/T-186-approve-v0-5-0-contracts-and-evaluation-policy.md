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

- Record explicit maintainer acceptance of P50-1 schemas/upgrades, comparison/check
  policies, finite limits, selection/batching, ownership/migration, curated approvals,
  receipts/handoffs and evaluation criteria.
- Resolve independent reference-judgment ownership, sample availability and justified
  minimum sizes/thresholds. Missing or unknown evidence keeps this gate blocked.
  Bind contract examples and feasibility review to exact revisions.
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
