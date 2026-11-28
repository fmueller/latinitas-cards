---
id: T-252-preserve-shared-source-evidence-for-repeated
title: Preserve shared source evidence for repeated phrase bindings
status: completed
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-249-evaluate-bounded-pydantic-ai-migration-feasibility
updated_at: "2026-11-28T01:50:27Z"
---

# T-252-preserve-shared-source-evidence-for-repeated Preserve shared source evidence for repeated phrase bindings

## Description

Correct the confirmed T250 installed-acceptance blocker: controlled phrases whose
bindings reuse a source entry abort before provider dispatch with duplicate
source_evidence. The defect also reproduces on 72fdf19 and predates F1/F2.
pipeline_orchestration._task appends identical unverified-phrase-entry evidence
per binding, which ModelTask rejects. See T250's acceptance notes and
specs/v0.5.0.md#whole-spec-acceptance-and-final-checks.

## Acceptance

- Same-entry verb-noun subject/object and noun-genitive bindings reach the
  provider path and produce explicit report outcomes rather than an envelope
  validation crash. Use synthetic German cards and a local endpoint only.
- Preserve every binding's role, form, sense and required source evidence;
  share identical evidence without weakening duplicate/corruption validation.
  Distinct-entry phrases and existing form behavior remain unchanged.
- Extend existing public-boundary regression coverage with representative
  same-entry cases; demonstrate failure before correction and success after.
- Return the corrected frozen candidate to the independent T250 owner for
  installed affected-scenario acceptance. This task alone does not complete A1.

## Verification Notes

Use implementing-changes. Run affected checks then the final Ruff, mypy and
full pytest chain before Taskrail verify/complete. No hosted inference, private
collection, policy enablement, import or publication. Commit/push main after
local gates without waiting for CI. Cite verification timestamps, not ignored
artifact paths. T250 stays blocked until its owner reruns affected acceptance.

## Implementation Notes

Fix: pipeline_orchestration._task now shares exactly identical Evidence and Scope
items from its binding loop (dict.fromkeys, first-seen order) before pattern and
context evidence is appended. Bindings that reuse one entry keep one confirmed
deck mapping per binding and one shared unverified-phrase-entry item; roles, forms
and senses stay in the request objective. ModelTask/Record duplicate validation is
unchanged. Regression: two same-entry German cases (verb-noun, noun-genitive from
phrase_fixture) extend test_phrase_saved_comparison_uses_independent_source_proofs
through a local synthetic endpoint. On the base both exit 1 with duplicate
source_evidence; with the fix both complete with an explicit unresolved comparison
(the fixture's senses are not Bennett-qualified). A base/head byte comparison of 35
form, Bennett and distinct-entry cards found identical tasks and scopes apart from
the usual config digest. Three wrong-fix mutations fail the new cases. Independent
medium-tier review (general, compatibility, test and Python lanes plus coordinator
disposition check) found no blocking issue. Deferred: a different-sense same-entry
test, and repeated identical reference contexts, which could still duplicate
context evidence (preexisting, reachability unverified).
- 2026-11-28T01:50:27Z: verification pass
