---
id: T-049-establish-reviewed-source-extraction-fixtures-and
title: Establish reviewed source extraction fixtures and supported layouts
status: todo
priority: high
spec_ref: specs/v0.2.0.md#evidence-led-source-extraction
dependencies: []
updated_at: "2026-10-04T23:05:46Z"
---

# T-049-establish-reviewed-source-extraction-fixtures-and Establish reviewed source extraction fixtures and supported layouts

## Description

Use existing skipped/ambiguous evidence to select a bounded set of additional source layouts. Publish reviewed sanitized examples before broadening extraction; do not invent supported layouts from speculation.

## Acceptance

- Inventory available v0.1 evidence and distinguish promised-layout regressions from new support. Missing evidence remains an explicit review requirement.
- Publish a supported-layout matrix and fixtures with independently reviewed expected candidates, role positions, alternatives, explicit omissions, formatting/separators, and embedded hints.
- Include similar-looking counterexamples that must remain unsupported or ambiguous; record exact withholding reasons.
- Include explicit fourth-role PPP versus supine distinctions and deponent/exceptional cases without claiming extraction proves their linguistic analysis.
- Preserve original text/provenance and document representative profile-confirmation examples; no private deck is required.
- Declare the reviewed sample, counting units and denominators, and whether categories overlap. Distinguish wholly skipped entries from generated entries with omitted/withheld-role warnings; overlapping counts must not appear to partition the sample. Do not claim universal coverage.

## Verification

Record review provenance and expected outputs suitable for subsequent regression tests. Fixture expectations must not be derived from the parser under test.
