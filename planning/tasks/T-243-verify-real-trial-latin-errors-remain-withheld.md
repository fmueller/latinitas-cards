---
id: T-243-verify-real-trial-latin-errors-remain-withheld
title: Verify real-trial Latin errors remain withheld after model-contract fixes
status: todo
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-242-use-provider-supported-structured-output-with
updated_at: "2026-11-27T01:28:42Z"
---

# T-243-verify-real-trial-latin-errors-remain-withheld Verify real-trial Latin errors remain withheld after model-contract fixes

## Description

Verify that fixing request shape and JSON reliability does not turn linguistically incorrect model agreement into accepted learning content. Use the observed trial mistakes as bounded development regressions, not protected evaluation data or a demand that all ten candidates pass.

## Acceptance

- Exercise wrong quantity in vias (vīās versus viās), the unsupported laudānt alternative versus laudant, and the unjustified restriction of puellae to nominative/vocative plural through source-bound public comparisons, with correct positive controls. Preserve valid ambiguity rather than forbidding all alternatives.
- Confirm malformed assertion keys, missing whole-card coverage and unsupported shared model claims remain rejected/withheld after T-241 and T-242. Fix any reachable in-scope defect found; do not hard-code these lemmas as special cases.
- Keep regression coverage small and representative; reuse existing tests and offline fixtures. Do not commit the private collection or responses, alter the user's collection, enable acceptance policy or claim a protected-study pass.
- Record what improved, which failures are model errors versus implementation defects, where the user gives card feedback, and the remaining release gates. Explain that schema adherence cannot establish Latin correctness.
- Prepare an exact-candidate ten-entry replay plan and preserve campaign accounting. Dispatch new hosted calls only with valid prospective permissions and within the existing campaign allocation; expired trial permissions are not reusable. An offline result must not be described as a fresh hosted success.
- Complete independent review and the required final validation chain; return the exact candidate for high-effort integrated adversarial QA.

## Verification Notes

- Trial diagnosis and ten candidate previews: https://ampcode.com/artifacts/ten-entry-latin-trial-408e5fb9-10c2-4fde-a753-08f91ae7f770 . Saved private receipts belong to the parent and need not be copied to task workers.

## Implementation Notes
