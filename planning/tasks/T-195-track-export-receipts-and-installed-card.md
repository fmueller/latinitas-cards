---
id: T-195-track-export-receipts-and-installed-card
title: Track export receipts and installed-card ineligibility
status: completed
priority: medium
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-194-explain-pipeline-outcomes-in-human-and-machine
updated_at: "2026-11-19T03:22:23Z"
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

- 2026-11-19T03:21:49Z: verification pass
- Added strict acceptance/assertion-bound export receipts, all-history invalidation
  accounting, persistent manual handoffs and whole-note sibling withholding.
  Unsupported lifecycle effects do not apply to Anki; destination inspection,
  full-tag/adoption approval and initial import remain separate.
- Final static gate: ruff and mypy passed; pytest had 3668 passed and 7 existing
  skips. Disposable real-CLI controls passed using synthetic engineering authority.
  Source/checker/policy changes, restored-history replay, corruption, historical
  reports, blank identities and curated ledger invalidation have regression coverage.
- Dedicated simplification and high-risk independent review completed. Six findings
  fixed and independently verified; one cumulative challenge round, no open findings.
  Native Anki resolution/history preservation, curated-consumer authority checking
  and release QA remain outer acceptance. No evaluation or policy enablement occurred.
- 2026-11-19T03:22:23Z: Task-local implementation verified at 2026-11-19T03:21:49Z; native Anki resolution and outer integrated acceptance remain separate
