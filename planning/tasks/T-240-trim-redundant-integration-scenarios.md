---
id: T-240-trim-redundant-integration-scenarios
title: Trim redundant integration scenarios to core behavior coverage
status: completed
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-239-optimize-evidence-validation-and-integration-ci-cost
updated_at: "2026-11-22T18:04:53Z"
---

# T-240-trim-redundant-integration-scenarios Trim redundant integration scenarios to core behavior coverage

## Description

The maintainer requests a Sol 6.1 high-effort implementation thread to actually
trim integration tests further, not merely analyze another speed optimization.
Keep representative happy paths and the most important edge cases; exhaustive
language/recipe/input combinations are not a requirement. Use engineering
judgment rather than interpreting all existing cases as permanent obligations.

This follows T239's parsing reuse and two-process integration execution. Accepted
main is c98c5e3c6379e710005846210105191e55f81486. T239's hosted integration
elapsed was 2169.21s on Python 3.13 and 2238.42s on Python 3.14, with 2909 nodes
and unchanged outcomes. Total hosted Build was 2423s. Those historical times
include uncontrolled runner variation; measure matched affected-suite results.

Review the expensive evidence/grammar/research workflows first. Identify tests
that repeat the same observable contract through unnecessary combinations or
rebuild expensive journeys to cover a small distinction. Delete or consolidate
redundant cases, retaining discriminating examples at the cheapest useful test
boundary. This is test reduction, not a product behavior or release-scope change.
The installed mutation adapter is narrow; use bounded deletion probes and explain
their scope, rather than launching a repository-wide mutation campaign. Scoped
setup/pruning permission already exists; no additional exhaustive framework is
requested. T200 stays blocked separately; this task does not bypass that gate.

Follow-up derived from T-239-optimize-evidence-validation-and-integration-ci-cost's verification or discovery.

## Acceptance

- Implement evidence-supported reduction in integration scenarios and report
  actual before/after collected cases and runtime. A reasonable smaller suite is
  the goal, not preservation of every existing permutation or an arbitrary quota.
- Keep public happy paths for forms, phrases, supported language behavior and
  saved reuse/export, plus important failure boundaries: contradiction/missing
  evidence, stale/tampered records and restoration, authority denial and complete
  outcome inventories. Reuse lower-level coverage instead of repeating the same
  failures for every language/recipe when those distinctions do not affect code.
- Record a compact keep/remove rationale and bounded deletion-probe/mutation
  evidence under the installed workflow. State what remains unmeasured; do not
  infer broad adequacy from the narrow research-reader adapter. No blanket deep
  mutation expansion or speculative regression tests to replace every deletion.
- Do not weaken product validation, policy controls or assertions merely to make
  the reduced suite pass. Do not introduce skips, quarantine or hidden selection
  filters. Preserve both Python versions and the existing CI failure gates.
- Complete scoped simplification, independent risk-scaled conventional review,
  dispositions and final required Ruff -> mypy -> pytest chain. Verify/complete
  with supported Taskrail, then commit/push directly to main under maintainer
  identity with installed hooks/no attribution. Subscribe to and supervise exact
  head Build/CodeQL, then return results to the parent without selecting a successor.
- No university-deck content, hosted inference, reference acquisition, study,
  policy enablement, native qualification, publication or deployment is included.

## Verification Notes

- Record final verification timestamps and evidence identities, not concrete
  gitignored planning artifact paths in committed notes.

## Implementation Notes

- Reduced the three affected files from 256 to 180 collected scenarios:
  meaning evidence 68 -> 30, research classification 16 -> 8, grammar pipeline
  172 -> 142. No new test journeys, skips or CI selection changes were added.
- Replaced language/family/recipe cross-products with representative tuples.
  Every phrase family, both languages and all three phrase recipes remain;
  each independent PPP opt-in and both source-bound/derived targets remain.
  Recipe rendering does not branch on phrase family, and target comparison
  uses the same exact-value/alternative guard for both kinds of target.
- Kept offline form/phrase paths plus every family's public comparison and
  saved reuse. Research keeps a form/de journey with all record, request,
  response and index tamper/restoration controls, plus a verb-noun/en journey.
  Classifier projection does not branch on language or phrase family.
- Consolidated the separate semantic-negative HTTP workflows into the existing
  research-negative journeys. Source/target labels, translation relation and
  reason, complete obligation inventory, two calls, export denial and saved
  proof reconstruction remain asserted without a second pipeline per case.
- Retained each distinct wrong-answer category and all four finite semantic
  dispositions. Removed mirrored locale cases, duplicate compatible-wording
  rejection, and repeated family context plumbing; bound/missing/changed
  context still runs for preposition-noun/en and verb-noun/de. Both languages'
  six present cells and locale-specific PPP meanings/drill labels remain.
- Left distinct authority, missing evidence, stale/tampered proof, recovery,
  invalidation and outcome-inventory boundaries intact. No further high-value
  reductions were identified in this bounded pass; untouched journeys were
  not exhaustively audited for consolidation.
- Independent medium-risk review covered general correctness, Python/test
  boundaries and rationale claims, with one fresh batched adversarial check.
  No required test fixes remained. Accepted sampling limits: noun-genitive/de
  and adjective-noun/en positive qualification, German experiential-compatible
  wording and German role-reversal wording are no longer individually pinned.
  Shared semantic categories remain tested, not these exact localized tuples.
  A vocabulary regression could lose diagnostic precision while still denying
  qualification/export. The explicit representative-sampling standard governs.
- Relocated-label validation: all five research-negative cases passed. In an
  isolated copy, changing independently grounded equivalent labels to
  unresolved failed the contradiction case at the source/target assertion
  (16.45s); restored source passed (17.04s). Owning production source did not
  change.
- Matched three-file commands on Python 3.14.7, one pytest process on this orb:
  accepted-base 256 passed in 3123.48s; final 180 passed in 2499.01s. Observed
  elapsed reduction was 624.47s (20.0%). Concurrent validation load differed,
  so this single local pair is not a controlled CI forecast.
- Final exact Ruff -> mypy -> full pytest chain passed on 2026-11-22,
  finishing at 18:04:04Z: Ruff clean, mypy clean on 249 source files,
  4001 passed/seven existing skips in 5108.91s. The skips are impossible
  annotation path-alias combinations, unchanged by this task. Reviewed and
  tested executable diff SHA256 against accepted c98c5e3:
  46c2f955600d7539452f48fa7e7b4f30135ba3fb899a4067293385f94d18b6b5.
  Subsequent edits are documentation and supported lifecycle metadata only.
- Cold finite-diagnostic deletion probe: eight original locale cases and four
  representative cases each passed twice. The selected relation function's
  32 mutants retained 28 killed/four survived, with no timeouts or changed
  scores; helper diff confirmed exactly four intended case removals. The
  other 315 generated mutants were deliberately not run. Four unchanged
  survivors expose this negative cohort's missing valid-equivalence and
  malformed-alternative checks, not proven equivalents. This supports only
  that locale deletion cohort, not the other sampled matrices or whole suite.
- Mutation only-tests guard reports the expected Taskrail STATE/task changes
  and adapter documentation; inspection confirms no production, dependency,
  executable configuration, resource ledger, release-gate or CI changes.
  Lifecycle metadata is validated separately. The original research-reader
  ledger is unchanged.
- 2026-11-22T18:04:53Z: verification pass
- 2026-11-22T18:04:53Z: Implementation verified and reviewed; authorized main delivery and exact-head CI supervision follow. Protected-custody study and release gates unchanged.
