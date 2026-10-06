---
id: T-072-label-unreviewed-fourth-role-as-linguistic-review
title: Label unreviewed fourth-role withholding as linguistic review
status: todo
priority: high
spec_ref: specs/v0.2.0.md#evidence-led-source-extraction
dependencies: []
updated_at: "2026-10-06T20:16:22Z"
---

# T-072-label-unreviewed-fourth-role-as-linguistic-review Label unreviewed fourth-role withholding as linguistic review

## Description

Found by the T-069 review. A cleanly extracted fourth role (e.g. `lātum`) withheld only because
no PPP/supine claim review exists is rendered as "(unresolved source evidence; target withheld)"
(generation.py marks comparison-withheld roles `unresolved`), and it produces no preview
warning because skips are computed before the comparison. specs/v0.2.0.md#evidence-led-source-extraction
requires distinguishing extraction failure from linguistic uncertainty in skip summaries and
review output. Release-blocking for T-068.

## Acceptance

- Withholding caused by missing linguistic review uses its own label/code (e.g. fourth-role review required), distinct from unresolved source evidence, in card answers and preview output.
- Preview counts such entries as generated-with-warnings, not as clean or as extraction failures.
- Update the pinned test in tests/unit/principal_part_generation_test.py and the extraction review count test; mandatory chain passes.

## Verification Notes

- Pending.

## Implementation Notes


