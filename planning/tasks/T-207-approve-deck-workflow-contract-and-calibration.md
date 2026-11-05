---
id: T-207-approve-deck-workflow-contract-and-calibration
title: Approve deck workflow contract and calibration revisions
status: blocked
priority: high
spec_ref: specs/v0.5.1.md#delivery-and-acceptance-plan
dependencies:
    - T-206-design-deck-workflow-follow-up-contracts
updated_at: "2026-11-05T22:11:32Z"
---

# T-207-approve-deck-workflow-contract-and-calibration Approve deck workflow contract and calibration revisions

## Description

Maintainer-only exact-revision approval gate. Remain blocked until the maintainer
decides; completing contract drafting is not approval.

## Acceptance

- Record explicit maintainer acceptance of exact contract and independently judged
  evaluation revisions, reference rights, source-stream authority and review-state
  migration. Unsupported/inconclusive calibration blocks dependent work.
- Resolve consequential choices before consumers and preserve T-186/T-205 authority.
  Planning-package approval does not approve these future design artifacts.
- Record schemas/upgrades, ownership and compatibility/native evidence requirements.
  Activation and publication are separate permissions. Do not self-unblock.

## Verification Notes

Inspect the maintainer decision and exact referenced revisions. A task status,
agent recommendation or draft completion is not a reviewer decision.

## Implementation Notes

- 2026-11-05T22:11:32Z: maintainer-only contract approval: accept exact follow-up schemas, source-stream authority and independently judged calibration revisions before consumers
