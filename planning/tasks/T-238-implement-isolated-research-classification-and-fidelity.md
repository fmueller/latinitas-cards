---
id: T-238-implement-isolated-research-classification-and-fidelity
title: Implement isolated research classification and fidelity validation
status: completed
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-237-produce-source-bound-translation-explanation
    - T-233-inventory-whole-card-evidence-producers-and
updated_at: "2026-11-21T15:58:19Z"
---

# T-238-implement-isolated-research-classification-and-fidelity Implement isolated research classification and fidelity validation

## Description

Implement proposal E in docs/grammar-pipeline.md, following the maintainer-approved
revision55 route to v0.5.0 readiness. T234-T237 now supply finite source-grounded
development evidence for complete bilingual cards. The application still has no
isolated research classifier that can measure candidate semantic sufficiency
without granting production acceptance authority.

Add that research-only boundary and independently qualify its fidelity to the
shared acceptance and recursive proof-reader semantics. Resolve the extraction/API
in the existing owners rather than creating a second, weaker acceptance engine.
This task prepares evaluation; it does not run or approve a held-out study.

Follow-up derived from T-233-inventory-whole-card-evidence-producers-and's verification or discovery.

## Acceptance

- Provide an explicitly versioned research classification record and a supported
  public caller/reader for would_accept, would_withhold and unscorable. Define and
  document their meanings, including missing evidence, processing failure,
  contradiction, ambiguity and unsupported scope. Retain the complete assertion
  inventory and individual check/applicability results and reasons, required
  coverage, alternatives, objections and failures; no failed slot may disappear
  from the denominator-facing record.
- Bind classification and reconstruction to exact candidate implementation,
  versions, policy scope/criteria, support registry, comparison, source/input and
  rights commitments, and actual saved blind invocation/output/stage evidence.
  Development criteria must be identified as development criteria, not approved
  statistical study thresholds. Altered, stale, missing or fabricated commitments
  fail closed; a stored would_accept flag is not authority or sufficient evidence.
- Share check-envelope, coverage, source-comparison and recursive proof-validation
  semantics with acceptance.assess/AcceptanceReader and orchestration readers.
  Isolate production authorization without bypassing or monkeypatching the
  installed registry, mutating its enabled cells, discarding unsupported siblings,
  treating unknown applicability as success, or taking model agreement as truth.
  Preserve current production behavior through any necessary refactor before
  adding research behavior.
- Exercise genuine finite source-grounded German and English complete-card
  positives through the supported public boundary, including both form and phrase
  coverage established by T237. Retain actual proposal-free captured requests and
  two-role proof journeys where required. Different-wording equivalents must not
  collapse to string equality; false or omitted meaning, roles, contexts and
  explanation siblings must still withhold. State the exact supported cells and
  reference limits; this is not a general-language or naturalness qualification.
- Independently validate per-assertion reasons and whole-card classifications on
  frozen positive, incomplete, contradictory, compatible/ambiguous, unsupported
  and processing-failure development cases. Include asymmetric stale/tampered
  saved request, output, binding and prerequisite-graph controls with restoration
  positive controls. Expected outcomes must be derived independently of the
  classifier. Synthetic enabled-policy fixtures alone do not qualify fidelity.
- Research records cannot become accepted decisions, human approvals, ordinary
  learning rows or import-ready handoffs. On the same positive evidence, stock
  disabled-policy preview, decide, replay/reuse and export still deny learning
  output. Exercise fabricated research records and attempted cross-store reuse;
  zero learning rows is the assertion, not necessarily absence of metadata files.
- Run a separate read-only, fixed-candidate installed/public-boundary checkpoint
  with source/wheel/installed identities, successful saved reuse before damage,
  ordinary export-denial controls and all material limitations disclosed. Reuse
  offline dependencies if needed, but distinguish that from a fresh dependency
  solve. Any synthetic accounting injection must be explicit and must not patch
  semantic validation, policy authority or the saved reader. Do not claim real
  hosted-model, native Anki, study or policy-enablement qualification from it.
- Complete red/green tests, dedicated simplification, independent risk-scaled
  conventional review and verified dispositions. Run the final unchanged exact
  Ruff -> mypy -> full pytest chain before supported Taskrail verify/complete.
  Preserve v0.4 characterization goldens and existing unsupported/export denials.
  Commit/push this task directly to main with repository hooks and maintainer
  identity; accept delivery only after exact-head Build and CodeQL succeed.
- Update existing documentation with the actual research contract, invocation,
  classifications, commitments, fidelity evidence and limits. Replace the stale
  absent-evaluator description without erasing prior producer/reference limits.
  Do not enable policies, invent curated approvals, restore/spend hosted campaign
  state, expose protected answers, start T200/T201, select a successor, publish or
  deploy. Report any genuine prerequisite defect instead of weakening acceptance.

## Verification Notes

- Record exact candidate/manifest identities, retained RED and failed runs,
  corrections, review coverage/dispositions and independent fidelity evidence.
  Cite Taskrail verification timestamps, not ignored artifact paths, in committed
  notes. Keep detailed temporary evidence under .amp/in/orb-backup/ or a thread
  artifact and distinguish overlapping test selections from additive totals.

## Implementation Notes

Accepted remote base: 2ba3e037cad4f08117ac61ff1dbabc29c169fa5a (T237),
with successful exact-head Build 44877891406 and CodeQL 44877891050.
Relevant owners: acceptance.py, acceptance_records.py, acceptance_policy.py,
pipeline_orchestration.py and its readers, semantic_comparison.py, and existing
public command/test boundaries. Inspect the actual source before choosing an API.

No hosted inference or new reference acquisition is allocated to this task.
Use retained lawful development references and synthetic local fixtures; request
prospective coordinator reconciliation if an indispensable resource is missing.
The approved global campaign remains four charged development attempts with all
reservations retained; source acquisition remains three charged requests. These
budgets must not be reset by this task or its checkpoint.
- 2026-11-21T15:57:56Z: verification pass
- 2026-11-21T15:58:19Z: Implemented version-1 research-only classify/save/read with shared evidence/proof validation and disabled production authority. Verification 2026-11-21T15:57:56Z: unchanged Ruff/mypy/full pytest passed (4076 passed, 7 skipped); independent installed checkpoint passed 22 bundles/120 slots/15 integrity controls with 11 overlapping zero-learning-row denial stores. F1/F2 corrected and independently verified; F3 refuted; simplifier no-op, high-risk review conditions closed. Retained original oracle/harness failures and prospective aggregate-rendered correction. Wheel SHA256 b26606b0805a4246bd7292955ceddbd6eccdbaed59dab2135f04a2d6aa1d3963; installed digest 4bb364f95f6bbcbf4814db7b39d7e49a1104360c48b94fa2b2465602ca7e9c58. Local synthetic HTTP only, amo1s/four exact bilingual phrase families, offline-resume reader-staleness limit retained; global hosted/reference totals unchanged at four/three. Implementation complete; push and exact-head Build/CodeQL delivery acceptance remain pending. No successor selected.
