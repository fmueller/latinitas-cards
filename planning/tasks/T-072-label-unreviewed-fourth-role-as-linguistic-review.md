---
id: T-072-label-unreviewed-fourth-role-as-linguistic-review
title: Label unreviewed fourth-role withholding as linguistic review
status: completed
priority: high
spec_ref: specs/v0.2.0.md#evidence-led-source-extraction
dependencies: []
updated_at: "2026-10-07T20:38:13Z"
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

- Verification run 2026-10-07T20:38:13Z: pinned v0.2.0; exact ruff/mypy/pytest
  chain passed (844 tests, 87 mypy files) after review and dispositions.
- RED/GREEN: asymmetric ferō/ferre/tulī/lātum answer wording failed before the
  fix. Independent Python finding T072-PY-01: "Completion-card prompts describe
  a cleanly extracted, linguistically withheld role as unresolved source evidence."
  Fresh candidate validation confirmed it. A sibling prompt assertion reproduced
  the failure; the card-rendering source of truth now separates linguistic
  withholding from parser uncertainty, including target/context eligibility.
- Dedicated code-simplifier reused the withheld set and simplified reason
  branches; 151 focused tests passed. Separate General/Python review lanes loaded
  mapped reviewer guidance. General: "No concrete task-relevant findings."
  Fresh disposition verification: "T072-PY-01 — RESOLVED" and "No concrete
  task-relevant findings." One cycle; no deferred or rejected findings.
- Literal generation counts: synthetic 13/25 generated, 12/25 wholly skipped,
  13 generated warnings; sanitized 3/5 generated, 2/5 skipped, 3 warnings (two
  linguistic, one unresolved extraction). Historical extraction strata retained.
- Inspected 2x Chromium reference-template/CSS renders for review-required,
  unresolved alternatives, and accepted fourth labels: four comparison rows,
  no clipping/overflow, distinct prompt/answer/context wording. Accepted labels
  still withhold unreviewed explanations. Browser evidence is not native Anki
  compatibility evidence; captures are linked in the execution report.
- No identity, recipe, slot, individual-claim binding, or fourth-role gate
  changes. No follow-up task or second implementation cycle.

## Implementation Notes

- 2026-10-07T20:38:13Z: verification pass
- 2026-10-07T20:38:13Z: Implemented distinct linguistic-review warning and rendered labels; independent finding fixed, final 844-test chain and browser reference checks pass.
