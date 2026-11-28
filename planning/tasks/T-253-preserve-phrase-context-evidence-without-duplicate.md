---
id: T-253-preserve-phrase-context-evidence-without-duplicate
title: Preserve phrase context evidence without duplicate-run crashes
status: completed
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-252-preserve-shared-source-evidence-for-repeated
updated_at: "2026-11-28T03:31:58Z"
---

# T-253-preserve-phrase-context-evidence-without-duplicate Preserve phrase context evidence without duplicate-run crashes

## Description

Correct T250 round 2's installed-acceptance blocker on 6aca005: identical
constructed-exercise-context values in a valid phrase reference produce duplicate
source evidence after T252's binding-loop deduplication. The provider run aborts
before dispatch without a report, including an unrelated clean phrase; offline
execution completes. See pipeline_orchestration._task and T250's notes, under
specs/v0.5.0.md#whole-spec-acceptance-and-final-checks.

## Acceptance

- Repeated identical phrase contexts reach explicit provider/report outcomes.
  In a mixed run, both the affected phrase and a clean sibling are processed.
- Preserve distinct context text, objective semantics and evidence scopes;
  exercise distinct contexts as a positive control, including equal-length text
  where scopes can coincide. Do not weaken strict duplicate/corruption validation
  or remove required evidence to bypass the failure.
- Retain T252's repeated-entry behavior, including distinct senses. Inspect the
  full evidence/scope assembly before ModelTask construction so this correction
  is not limited to another intermediate insertion point.
- Extend existing public-boundary tests with focused red/green regression
  coverage using synthetic German data and a local endpoint. No broad new matrix.
- Return the corrected candidate to T250's independent installed acceptance
  owner; T253 completion alone does not close A1 or unblock release gates.

## Verification Notes

Use implementing-changes. Run affected checks during development, then final
Ruff, mypy and full pytest before Taskrail verify/complete. No hosted inference,
private collection, replay, policy enablement, import or publication. Commit/push
main after local gates without waiting for CI. Cite verification timestamps,
not ignored artifact paths. T250 remains pending independent acceptance.

## Implementation Notes

Fix: pipeline_orchestration._task now applies T-252's exact-identity sharing
(dict.fromkeys, first-seen order) once, after the whole evidence/scope
assembly and immediately before ModelTask construction, instead of after the
binding loop. Pattern and context items are now covered too. Every context
stays verbatim in the request objective; distinct contexts keep separate
evidence; ModelTask/RequestContext duplicate validation is unchanged. A second
base failure surfaced while reproducing: two distinct equal-length contexts
produce identical context Scopes, and RequestContext rejected them with
`duplicate scope`, also aborting the run. The same move fixes it. Tasks the
base accepted are byte-identical, so saved calls, scopes and permissions keep
their digests.

Regression: one German noun-genitive case (contexts A, A, B with B distinct
but equal length, plus a clean sibling phrase) in phrase_evidence_test through
the local synthetic endpoint. On the base it exits 1 with duplicate
source_evidence and nothing is dispatched; with the fix the run completes, both
cards dispatch, and one context scope plus [A, B] evidence is sent. Wrong-fix
mutants (evidence-only late sharing, per-text context dedup, dropped context
scopes) all fail it. Tests added: 1; removed: 0.

Review: simplifier (no code changes; comment wording corrected), high-tier
conventional review relayed as four lanes plus an independent adversarial
verifier; no blocking findings. Accepted: assert scopes, collapse to one case.
Gap: same-entry different-sense binding remains covered by T-250's installed
rerun, not by a committed test.
- 2026-11-28T03:31:58Z: verification pass
