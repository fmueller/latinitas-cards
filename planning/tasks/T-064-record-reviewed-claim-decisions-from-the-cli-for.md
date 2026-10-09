---
id: T-064-record-reviewed-claim-decisions-from-the-cli-for
title: Record reviewed claim decisions from the CLI for all generated cards
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#reviewed-claim-decisions-for-all-generated-cards
dependencies:
    - T-068-publish-v0-2-0
updated_at: "2026-10-06T18:13:52Z"
---

# T-064-record-reviewed-claim-decisions-from-the-cli-for Record reviewed claim decisions from the CLI for all generated cards

## Description

Add a non-interactive, agent-friendly CLI route to list open linguistic claims and
record, correct or withhold individual decisions in a versioned sidecar, consumed by
preview/export for every profile and recipe, following
`specs/v0.2.1.md#reviewed-claim-decisions-for-all-generated-cards`. Today only
form parsing accepts decisions (hand-edited JSON); principal-part labels, splits and
explanations need the Python API. Pair it with read-only inspection and a check mode
so agents can verify generated cards in a loop without starting Anki.

## Acceptance

- List open claims per source/profile/recipe with identity, form, proposal,
  alternatives, evidence, status and fingerprint, as text and JSON.
- Record `accepted`/`withheld` decisions with reviewer and reason, singly or as a JSON
  batch; no approve-all path. Reviewer-authored values become new claims with their
  own evidence and review.
- Validate batches atomically before writing: any conflicting, stale or invalid item
  rejects the whole batch with nonzero exit and item-level JSON diagnostics, leaving
  the sidecar byte-unchanged. Test a valid acceptance mixed with each failure class.
- Principal-part export/preview and form-parsing read the sidecar automatically; stale or
  unmatched decisions are reported and ignored.
- Accepted claims appear on exported cards for both principal-part recipes and parsing;
  withheld/missing ones do not. Reruns with the same sidecar give identical CSV and
  unchanged note/card identities.
- Read-only inspection of source+profile+sidecar, exported CSV or `.apkg`/`.colpkg`
  shows per card the identity, slot, eligibility, plain-text prompt/answer and backing
  claims with reasons for withheld/absent content (text and JSON), without Anki.
- A check mode exits nonzero on unreviewed asserted content, stale decisions, drifted
  fields or identity/slot mismatches; passes on a fresh export.
- Offline package inspection supports documented Latinitas template/layout versions,
  deriving plain-text prompts/answers from their known fields/templates, not emulating
  arbitrary client JavaScript or cloze/conditional templates. Unsupported layouts are
  diagnosed and fail check mode; include supported and unsupported package fixtures.
- Tests cover agent batches, conflicts, authored corrections, profile changes and stale
  decisions.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- This task owns the shared sidecar/CLI/inspection contract and adapters for both
  principal-part recipes and form parsing. T-084/T-087 add phrase inputs and recipe
  adapters later; completing this foundation does not claim phrase coverage.
- Fingerprints bind form, context, evidence, alternatives and linguistic profile,
  not presentation. Decisions are attestations, not export or managed-plan approval.
  Missing forms require separately reviewed source corrections, never acceptance
  of a nonexistent form. Sidecars remain separate from destination state.
- Inspection/check is offline, read-only and available in text/JSON. Package checks
  compare supplied artifacts with current source/profile/sidecar; they do not
  require this release to generate packages or run Anki.
- Implement in reviewable milestones: shared decision persistence, existing-recipe
  adapters, then inspection/check. Keep one shared contract for later phrase adapters.
