---
id: T-195-track-export-receipts-and-installed-card
title: Track export receipts and installed-card ineligibility
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-194-explain-pipeline-outcomes-in-human-and-machine
updated_at: "2026-11-02T23:10:11Z"
---

# T-195-track-export-receipts-and-installed-card Track export receipts and installed-card ineligibility

## Description

P50-5 receipts/lifecycle handoff, also consumed by curated withdrawal. Own export
receipts and supported retained-field managed/manual-resolution output. Follow
approved identity/ownership contracts and spec execution/validation requirements.

## Acceptance

- Bind exported semantic note/card identities to assertions, policies and evidence.
  Source/checker changes or policy withdrawal list installed affected identities and
  reasons; new exports do not assert stale acceptance.
- Emit required manual actions for unsupported effects or a supported retained-field
  managed plan. Omission does not remove assertions, badges or cards from Anki;
  never claim CSV suspension/retirement.
- Preserve user fields/tags and scheduling/history. Destination inspection/approval
  and initial import remain separate; no direct Anki integration.
- Report unresolved actions without claiming application. Include curated approvals
  withdrawn/invalidated as well as automatic policy withdrawal.

## Verification Notes

Use disposable receipts/destinations: export, invalidate source and policy separately,
inspect affected identities and field diffs. Reject unsupported lifecycle effects and
unauthorized writes. Native resolution is integrated acceptance. Run the mandatory chain.

## Implementation Notes
