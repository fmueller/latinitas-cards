---
id: T-052-generate-supported-principal-part-relationships
title: Generate supported principal-part relationships and explanations
status: completed
priority: high
spec_ref: specs/v0.2.0.md#calibrated-form-parsing
dependencies:
    - T-050-extend-source-extraction-for-the-reviewed-layouts
    - T-051-establish-claim-level-evidence-and-calibrated
updated_at: "2026-10-05T02:04:35Z"
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

## Implementation and workflow evidence

- Baseline: fetched origin/main and confirmed containment of accepted T-048–T-051
  changes. Taskrail validate passed; next --json selected this task under pinned
  specs/v0.2.0.md, with completed T-050/T-051 dependencies. No source-main drift
  was transferred. Started through Taskrail at 2026-10-05T01:32:05Z.
- New focused principal_relationships module returns linguistic comparison data
  independently of HTML: four roles, presence/withholding, separately reviewed
  segmentation and German explanation, and claim evidence/decisions. Generation
  binds proposals to scoped LatinitasID, actual text/profile/extraction; no
  automatic analyzer, calibration threshold, suffix inference or new recipe.
- Fourth forms require individually accepted PPP/supine context labels. Older
  unreviewed fixtures now retain six rather than eight sibling cards. Conflicting
  PPP/supine profiles require explicit resolution; stale reviews cannot silently
  reinterpret earlier roles. Existing checkpoint eligibility-loss safeguards
  remain in force for previous exports.
- Generation/typed preview/export share review inputs and escaped evidence.
  Answer fields provide focused explanation, compact accepted related splits and
  a basic collapsed four-role handoff. Unsupported gaps remain explicit.
  T-053 retains subsequent presentation/styling ownership; cli.py is unchanged.
- RED: new domain tests failed with ModuleNotFoundError before implementation.
  Generation regression then failed with 8 cards instead of expected 6; GREEN
  after review-gated fourth-role generation. Additional RED/GREEN covered
  competing accept/withhold decisions, incompatible profiles, stale-candidate
  review visibility and compact related-stem withholding.
- Initial exact ruff/mypy/pytest -v chain passed: 72 source files, 688 tests.
  Dedicated code-simplifier Task loaded the personal skill, removed only a
  redundant test import, and passed 151 focused tests. No suggestion rejected.
- Independent parallel code-reviewer Tasks: General (ECC code-reviewer; no
  companion), Python (python-reviewer and python-patterns), Security
  (security-reviewer and security-review). Checks: 160, 151 and 70 focused tests
  respectively. General and Security: "No concrete task-relevant findings."
  Security was selected for HTML/terminal source-evidence trust boundaries;
  Database, framework and ML lanes omitted because no database/network/model
  implementation or framework contracts changed. Specialist budget not exceeded.
- Fresh candidate-validation confirmed F1, quoted below. Fixed with strict TDD:
  two analysis-conflict tests failed on role-wide withholding, then passed with
  per-kind gating; independently accepted siblings remain eligible.
- Fresh disposition verification resolved F1 and raised F2. Fresh candidate
  validation confirmed F2, quoted below. Dedicated None/None regression failed
  because segmentation was asserted, then passed after requiring an actual
  extraction snapshot. Positive synthetic fixtures now explicitly carry snapshots;
  the general T-051 Claim API remains unchanged.
- Second/final fresh disposition-verification Task loaded code-reviewer, verified
  both fixes, passed 16 focused tests and reported "No new concrete task-relevant
  findings." No rejected candidates, deferred findings or unresolved findings.
- Fresh exact chain after fixes: ruff passed; mypy passed on 72 source files;
  pytest -v passed 691 tests. Full-chain restart after earlier mypy variable-name
  failure. Taskrail verification/completion follows the final fresh chain.
- Manual Chromium check of actual generated fields with existing reference CSS:
  2x collapsed and full expanded captures inspected in the implementation thread;
  Latin-first prompts, accepted detailed split/German prose/related infinitive,
  four named roles, Supinum em-dash withholding and unreviewed-analysis notices.
  DOM confirmed two initially closed comparisons, both opened successfully, and
  withheld fourth roles in both. Existing oversized/bold reference typography is
  reserved for T-053, not silently restyled here. This is browser field rendering,
  not native Anki or Latin-expert calibration evidence.

## Verbatim validated findings and dispositions

F1 — medium, Python/domain; principal_relationships.py accepted-value grouping and
role-wide blocked flag:

> A conflict in one claim kind suppresses independently accepted claims of the
> other kind for the same role, contrary to the requirement that segmentation and
> explanation be assessed independently.

Fixed: only label conflicts block the role; segmentation and explanation each
retain independent singleton/withholding gates. Independently expected regression
checks both conflict directions. Fresh review verified the disposition.

F2 — medium, Python/edge-case; principal_relationships.py extraction comparison:

> Claims can be accepted without being bound to any source extraction when both
> the claim and parsed input have extraction=None.

Fixed: absent parsed extraction always withholds a proposal; exact matching
snapshots are still required. Dedicated negative regression and positive snapshot
fixtures pass. Fresh review verified the disposition. Two total review/fix cycles.

## Scoped remaining work

T-053 owns finer answer-side presentation, typography and interaction polish.
The explicit review path remains the offline Python API; no persistent CLI review
editor/importer or domain calibration is claimed. Tests are independently authored
synthetic expectations, not production analyzer observations or Latin-expert
approval. No new follow-up task was warranted beyond already tracked T-053.

## Implementation Notes

- 2026-10-05T02:04:35Z: verification pass
- 2026-10-05T02:04:35Z: Reviewed offline claim handoff implemented; automatic calibration disabled; T-053 retains presentation scope. Final full chain passed 691 tests and both validated review findings are fixed.
