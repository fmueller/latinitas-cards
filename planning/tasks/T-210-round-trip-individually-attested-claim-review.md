---
id: T-210-round-trip-individually-attested-claim-review
title: Round trip individually attested claim review sheets
status: todo
priority: high
spec_ref: specs/v0.5.1.md#individual-review-round-trip
dependencies:
    - T-209-generate-reviewed-nominal-completion-and
    - T-187-implement-versioned-automatic-decisions-and-reuse
updated_at: "2026-11-05T22:11:24Z"
---

# T-210-round-trip-individually-attested-claim-review Round trip individually attested claim review sheets

## Description

Deliver FR-3 claims export-review/import-review through existing shared decisions
and the approved CSV contract. Own interchange before calibration consumers.

## Acceptance

- Export lemma/form/proposed role or value/evidence/fingerprint with empty verdict,
  reviewer/reason cells. Import only explicit accept/reject plus reviewer; blank
  rows stay open, no approve-all. Use internal mapping/default audit reasons.
- Mark 10 accept, 2 reject, rest blank: exactly 12 individual decisions. Identical
  reimport is idempotent. Untrusted/formula cells cannot alter canonical bindings.
- Stale/unknown fingerprints, edited bound evidence, duplicate/conflicting decisions,
  invalid verdicts or missing reviewers fail atomically with row diagnostics.
- Accepted genuine PPP labels unlock eligible PPP cards; supine-only, missing and
  future-participle forms never become PPP. Preserve legitimate supine cards and
  fail-closed source/calibration drift checks.

## Verification Notes

Exercise public CLI positive/negative round trips and inspect unchanged blank rows
and store invariants. Run mandatory ruff → mypy → pytest chain; follow the full
spec compatibility, evidence and permission requirements.

## Implementation Notes
