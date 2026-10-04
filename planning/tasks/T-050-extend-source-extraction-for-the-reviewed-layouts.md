---
id: T-050-extend-source-extraction-for-the-reviewed-layouts
title: Extend source extraction for the reviewed layouts
status: todo
priority: high
spec_ref: specs/v0.2.0.md#evidence-led-source-extraction
dependencies:
    - T-049-establish-reviewed-source-extraction-fixtures-and
updated_at: "2026-10-04T23:05:46Z"
---

# T-050-extend-source-extraction-for-the-reviewed-layouts Extend source extraction for the reviewed layouts

## Description

Implement only the additional extraction rules justified by the reviewed fixture set, using the existing confirmed-profile workflow.

## Acceptance

- Preserve original text and provenance alongside normalized candidates and applied-rule records.
- Keep alternatives unresolved until an explicit selection rule or review; retain omissions in their positions rather than shifting later forms. Preserve mixed hints as evidence, not grammatical assertions.
- Unknown layouts and conflicting hints produce actionable unsupported/ambiguous outcomes.
- Profile confirmation and skip/review output distinguish extraction failure from linguistic uncertainty. Report reviewed/generated/skipped/ambiguous counts using the declared units, denominators, and overlap rules; distinguish wholly skipped entries from generated entries with omitted/withheld-role warnings.
- Exact-output tests cover every accepted layout and its counterexamples, especially omitted middle roles and alternatives. Existing supported layouts remain regressions.
- Preserve evidence for the claim-policy handoff: unresolved alternatives and conflicting role evidence must remain identifiable in generation/preview/export, not silently become unconditional facts in either existing recipe. Verify that handoff once claim-policy and principal-part-analysis are available.

## Verification

Use red/green tests for new behavior. Run uv run ruff check, uv run mypy, and uv run pytest -v in that order; after any failure rerun the full chain.
