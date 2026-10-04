---
id: T-052-generate-supported-principal-part-relationships
title: Generate supported principal-part relationships and explanations
status: todo
priority: high
spec_ref: specs/v0.2.0.md#calibrated-form-parsing
dependencies:
    - T-050-extend-source-extraction-for-the-reviewed-layouts
    - T-051-establish-claim-level-evidence-and-calibrated
updated_at: "2026-10-04T23:05:46Z"
---

# T-052-generate-supported-principal-part-relationships Generate supported principal-part relationships and explanations

## Description

Produce reviewable principal-part relationships and focused German explanations for the existing completion and recognition recipes before introducing token parsing exercises. Keep linguistic data independent of rendering.

## Acceptance

- The confirmed profile identifies the fourth role explicitly as PPP or supine; do not infer PPP, case, or gender from an uncontextualized -um. Resolve incompatible profile changes explicitly, not by silently reinterpreting older profiles.
- Detailed stem/formation-marker/ending segmentation is emitted only for supported accepted claims; coarse stem/ending analysis remains possible when it alone is justified.
- Compare irregular relationships without fabricated character derivations. Ambiguous, omitted, exceptional, and deponent cases retain alternatives or actionable withholding outcomes.
- Provide a focused German explanation of the tested form and compact related-stem comparison, plus data and evidence-backed reviewable further explanation for the complete four-role comparison with explicit absent/withheld roles. Unreviewed prose cannot fill explanatory gaps.
- Preserve existing Latin-first prompts, recipe keys, object identities, and template bindings. No new principal-part recipe is added.
- Generation and preview/export tests prove unresolved alternatives, conflicting role evidence, and withheld claims cannot become unconditional facts through either existing recipe. Preserve eligible unaffected sibling cards.
- Independently expected tests cover detailed, coarse, irregular, ambiguous, omitted, PPP, supine, and deponent cases; changing styling cannot change a claim.

## Verification

Use red/green tests. Run the mandatory ruff, mypy, pytest -v chain; review linguistic outputs against the evaluated claim policy.
