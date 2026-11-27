---
id: T-243-verify-real-trial-latin-errors-remain-withheld
title: Verify real-trial Latin errors remain withheld after model-contract fixes
status: completed
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-242-use-provider-supported-structured-output-with
updated_at: "2026-11-27T05:57:36Z"
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

- Added six source-bound public workflow controls for the three reported Latin
  mistakes and correct counterparts, plus one nested assertion-key rejection case;
  no tests removed. Existing fixtures and test files are reused.
- No reachable implementation defect was found. Correct and erroneous model
  agreement remains withheld without qualified whole-card evidence. These cases
  do not independently diagnose each Latin error or qualify the models. Preview
  retains all four supplied puellae readings; saved checks retain laudānt.
- Existing supported target/stem, missing-sibling, surface and alternative controls
  remain distinct from these unsupported source/rendered semantic paths. No
  production code, policy, private input, hosted call or real ledger was changed.
- Dedicated simplifier, three independent medium-risk review lanes, fresh batched
  challenge and coordinator disposition verification completed without required
  findings. Optional retained-alternative assertion and feedback/replay wording
  clarifications were accepted and rechecked.
- Runtime-only fault injection dropping alternatives failed at the new assertion
  in 5.44s; normal targeted execution passed in 5.54s. Initial fixture/harness
  mistakes were not product RED evidence. Six controls passed in 30.73s; the
  broader targeted selection passed 24 cases in 108.80s.
- Final exact Ruff -> mypy -> full pytest chain started 2026-11-27T05:14:05Z:
  Ruff passed, mypy passed for 249 files, pytest passed 4020 tests with 7 existing
  skips and 1 serializer warning in 2565.48s; total chain 2567s. Tested executable
  diff SHA-256: de92efdbc1740e95eeb87ef1a9a9fab16af54af33e32fa90664f65cc96d9c8ff.
  No controlled runtime delta or speed claim follows.
- Mutation adapter doctor passed; production changed scope is empty, so no score
  is claimed. Comment-hygiene AST tooling was unavailable (uncomment absent);
  no new code comments were added. RTK 0.51.0 wrappers/proxy were used.
- Existing grammar-pipeline documentation records card-feedback guidance and the
  exact-candidate same-ten-entry replay procedure. The parent owns independent
  high-effort integrated QA, fresh hosted permission, unchanged campaign accounting
  and regenerated trial report. Offline checks do not prove live correctness or
  protected-study qualification. Policy, native acceptance and publication gates
  remain separate; no successor was selected.
- 2026-11-27T05:57:36Z: verification pass
