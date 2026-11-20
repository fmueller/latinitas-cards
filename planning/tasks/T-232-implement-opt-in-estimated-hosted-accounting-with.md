---
id: T-232-implement-opt-in-estimated-hosted-accounting-with
title: Implement opt-in estimated hosted accounting with shared campaign limits
status: completed
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-189-add-explicit-providers-and-blind-linguistic
updated_at: "2026-11-20T15:55:10Z"
---

# T-232-implement-opt-in-estimated-hosted-accounting-with Implement opt-in estimated hosted accounting with shared campaign limits

## Description

P1 of the maintainer-approved revision 55 MiniMax release-readiness plan:
https://ampcode.com/artifacts/v0-5-0-unblocking-plan-d8da3973-9b6a-451c-8ce1-93f5c4004857.
The maintainer ordered execution under its limits and stop before publication.
Implement the adopted opt-in estimated accounting/permission/campaign contract
under specs/v0.5.0.md#cli-orchestration-and-provider-boundaries before its consumers.
This is a T-189 follow-up, not MiniMax runtime qualification or empirical evaluation.

Owning scope: providers, pipeline profile/orchestration, versioned permission,
report/provenance schemas and their documentation. Use the SAME supported pipeline,
not a parallel general-purpose development runner. No hosted dispatch belongs to
P1: use fakes and disposable local HTTP only. T231 remains blocked for P1 and actual
mode-specific runtime qualification under P2. Its two-probe allocation stays closed.

## Acceptance

- Guaranteed accounting remains the default, including existing fail-closed behavior
  when no proven complete-service-input bound is installed. Preserve token_bound's
  proven-bound semantics. Estimated mode is a separately named, explicit opt-in
  contract for development, isolated evaluation and eventual qualified normal
  runtime; no heuristic is passed off as a proven bound and no silent fallback
  selects a provider or accounting mode. Existing offline zero-call reuse remains.
- Version permissions, reports and provenance to record accounting kind, purpose,
  estimator/configuration identity, exact serialized-request commitments, full
  reservations and actual provider-reported usage separately. Hosted permission
  binds endpoint/model, exact approved input manifest, purpose, effective limits,
  estimator, expiry and possible existing Credits/balance use. Preserve scope and
  credential redaction; no account changes, top-ups or new subscriptions. Document
  that estimated service-token totals and charges are not guaranteed ceilings.
- Implement a durable enclosing campaign allocation shared across runs and proposal/
  counter-check roles, not replenished by a new run, copied directory, retry or
  renewed permission. Whole-plan ceilings are 2000 dispatched attempts, 40000000
  estimated input-plus-reserved-output tokens, USD 25 estimated public-rate valuation
  and 24 hours cumulative provider-execution time. Phase attempt caps are development/
  output-feasibility 80, protected study 1600, real/native acceptance 160 and
  adversarial QA/fix verification 160. Fixes consume the same allocation; unused
  phase capacity cannot be silently reassigned. T231's closed slots do not transfer.
- Before dispatch reserve against the strictest remaining permission, run, phase
  and campaign limits, stopping at the first applicable ceiling. Preserve existing
  per-run 200-attempt/200000-token limits, explicitly estimated in estimated mode.
  Every complete serialized request is at most 16 KiB, requests at most 2048
  generated tokens including reasoning, has a 60-second deadline and runs alone.
  Enforce data/byte/output/time/rate guards across the enclosing allocation; no
  hidden API retries or concurrent runs that bypass one-in-flight accounting.
- Use an explicitly versioned padded estimate over exact serialized bytes; the
  historical 2 x UTF-8 bytes + 8192 input plus full output reservation is only a
  candidate heuristic. Check applicable public pricing before execution; missing
  rates block spending, not invented USD valuations. Missing/invalid usage or usage
  above reservation stops further dispatch for review. Preserve full reservations
  after every outcome, with separate usage/pricing observations and no refunds;
  report actual plan deduction as known or unknown rather than inferring billing.
- Durably prepare reservations and request/scope commitments before sending; retain
  spending across crash/reopen, permission renewal, cancellation, failure and unknown
  outcomes. Unknown dispatches stay charged and block further work until the existing
  explicit reconciliation/abandonment path is followed. Potential duplicate
  processing requires explicit acknowledgment and fresh remaining budget. Map ALL
  error labels conservatively into a closed outcome model; do not repeat the old
  scratch runner's unknown-label omission. Cancellation does not undo remote billing.
- Record model aliases/service changes; an unpinnable deployment revision is an
  explicit reproducibility limitation, never a fabricated immutable version. A
  detected material service change invalidates affected reuse and requires
  requalification. Preserve proposal-free blind input, untrusted-source/model-text
  handling, credential confidentiality, exact scope, retained response/error evidence
  and redaction.
- Accounting mode grants no policy, semantic, whole-card or export authority. Keep
  development captures separate from reference truth and isolated evaluation
  would-accept artifacts separate from ordinary learning exports. Production
  eligibility still needs actual semantic checks and separately enabled policy;
  estimated spending alone neither grants nor removes linguistic authority. Do not
  run references, protected studies, qualification, T200/T201 lifecycle or publication.
- Fake/local-HTTP boundary tests prove exact/one-over call, byte, estimate and
  deadline refusal; shared-role/cross-run/campaign and phase spending; crash/reopen;
  unknown/duplicate/cancellation controls; usage/pricing stops; redaction; unchanged
  guaranteed-mode refusal; and no acceptance/export bypass from accounting metadata.
  Independently execute the installed pipeline slice with positive/negative controls.

## Verification Notes

The future P1 owner must load implementing-changes and the actual autonomous-backlog
skill, reconcile this unpublished task/T231 handoff with fresh main, validate and
require authoritative selection with off_spec=false before starting. Own only P1.
Use red/green TDD for changed behavior, dedicated simplification, independent
risk-scaled conventional review and verified finding dispositions, then an independent
bounded runnable-slice checkpoint in that checkout. Preserve all failed attempts,
prior review/campaign budgets and exact executable commitments; fake/local-HTTP
evidence is not hosted qualification.

Before verify/complete or code delivery, freeze final executable bytes and run this
exact chain in order: uv run ruff check -> uv run mypy -> uv run pytest -v. Run
applicable Taskrail, whitespace and policy guards too. Only genuinely completed P1
may use supported verify/complete and authorized commit/push directly to main with
hooks, maintainer identity and no attribution trailers. Subscribe to exact-head CI;
substantive delivery requires Build and CodeQL success. Retain the checkout for
corrections. Do not release, publish, deploy or choose a successor.

## Implementation Notes

- Administrative materialization only. This draft was not started, implemented,
  verified, completed, committed or pushed by the T231 owner. No P1 hosted call is
  assigned or permitted. Taskrail write ownership transfers only after the complete
  unpublished T231/P1 task bundle is delivered and reconciled by the receiving owner.
- 2026-11-20T15:55:01Z: verification pass
- 2026-11-20T15:55:10Z: Implemented P1 only; verification pass at 2026-11-20T15:55:01Z after independent review, installed checkpoint and exact final Ruff/mypy/pytest chain (3874 passed, 7 existing skips). Preserve blocked T231 history and supported handoff reconciliation; its closed probe allocation does not transfer. No hosted inference (0/2000), qualification, policy enablement, protected study, T200/T201 lifecycle, successor selection or publication. Direct-main delivery and exact-head Build/CodeQL acceptance are separate pending gates.
