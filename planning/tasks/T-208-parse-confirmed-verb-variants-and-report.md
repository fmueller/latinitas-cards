---
id: T-208-parse-confirmed-verb-variants-and-report
title: Parse confirmed verb variants and report exhaustive skips
status: todo
priority: high
spec_ref: specs/v0.5.1.md#verb-parsing-and-skip-reports
dependencies:
    - T-207-approve-deck-workflow-contract-and-calibration
updated_at: "2026-11-05T22:11:24Z"
---

# T-208-parse-confirmed-verb-variants-and-report Parse confirmed verb variants and report exhaustive skips

## Description

Deliver FR-4 by extending source_extraction/principal_parts and focused command
modules, not growing cli.py. Own extraction consumers before nominal routing.

## Acceptance

- Confirm policies for trailing usage HTML, compound versus alternative pipes,
  within-role slash alternatives and deponent/semi-deponent layouts. Preserve
  raw provenance/display markers, absent roles and source bytes; invent no forms.
- Assert expected parsed role sequence, alternatives and usage separation for
  timēre HTML, ad|esse, placēre alternatives, esse usage hints, NBSP, missing forms
  and deponent/semi-deponent fixtures even when linguistic review withholds cards.
- Reject ambiguous boundaries/conflicting pipe policies. Do not turn nominal
  prae|ceptum into a verb. Preserve old-profile output and fourth-role review.
- Expose complete preview --json skips and a readable lemma/source-key/reason/
  fragment table; label truncation, offer full output and separate parse success
  from eligibility counts.

## Verification Notes

Use contrasting fixtures via preview/generate, compare source checksums and
unchanged-profile goldens. Run uv run ruff check → uv run mypy → uv run pytest -v;
restart the full chain after failure. Follow all spec compatibility/permission rules.

## Implementation Notes
