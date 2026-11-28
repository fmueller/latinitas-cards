---
id: T-256-bounded-model-retries
title: Add bounded two-retry recovery for transient model failures
status: completed
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-254-replay-approved-ten-german-cards-after-post-trial
updated_at: "2026-11-28T14:33:39Z"
---

# T-256-bounded-model-retries Add bounded two-retry recovery for transient model failures

## Description

Implement the maintainer-requested simple model-call retry policy: up to two
retries after the first attempt, with bounded waiting for transient failures.
Use implementing-changes and keep prompt simplification owned by T257.

## Acceptance

- Inspect existing retry paths first; providers.py already has a bounded loop and
  a transient retry branch that excludes accounted calls. Reuse existing owners.
- At most three actual attempts per logical request; use bounded backoff/jitter
  and respect cancellation, deadlines and provider retry guidance where applicable.
- Every attempt requires valid permission, prospective budget and durable retained
  accounting. Unknown outcomes must not silently redispatch or refund reservations.
- Do not retry permanent authentication/configuration failures or silently repair,
  coerce or approve malformed answers. Preserve strict parsing and acceptance gates.
- Reconcile affected spec/contract language explicitly with this new authorized
  policy; no SDK-hidden retries, unbounded loops or provider fallback.
- Implement with synthetic/loopback tests only; no live MiniMax or other provider
  inference is authorized for implementation. Complete required local gates,
  independent review and Taskrail lifecycle; commit and push on main when ready.

## Verification Notes

- Red/green focused tests must distinguish two retries from one or three and
  cover transient recovery/exhaustion plus nonretryable and budget/unknown cases.
  Run the final repository gate on the delivered executable state. Do not wait
  for CI before the sibling task can proceed; subscribe to exact-head results.

## Implementation Notes

- Reused the existing ProviderRun.request attempt loop; no new retry layer.
  `Limits.retries` and pipeline `processing.retries` now accept 0–2 (default
  stays 1). Estimated-accounting profiles and MiniMax runtime admission no
  longer pin zero retries, so at most three attempts per logical request.
- Backoff before retry n is n + uniform(0, 0.5) seconds through the
  cancellation event (at most about 4 s per request). Each retry rechecks
  cancellation, permission and every call/token/campaign limit, journals a new
  call and takes a fresh full reservation; failed attempts stay charged.
- Only `transient` retries. HTTPTransport now connects explicitly: a refused,
  reset or timed-out TCP connect or TLS handshake (no HTTP byte sent) is
  transient; DNS and TLS protocol/certificate errors are non-retryable
  `failed`. HTTP errors including 401/403/429/5xx, post-send timeouts,
  disconnects and cancellation stay `unknown` and never retry, so Retry-After
  is not consulted. A transient attempt that overran its deadline is recorded
  `unknown` without retry. Malformed answers and contract violations are not
  retried or repaired. Transient not-sent attempts no longer raise a usage
  review stop. Spec, provider-boundary, contract proposal and changelog
  updated. Prompt builders (T257) untouched.
- Evidence: red/green on the new focused tests; seven wrong-implementation
  mutants (fixed backoff, one/three retries, usage stop on transient,
  refused→failed, TLS→transient, no accounted retry) each fail a test.
  Simplifier pass; independent high-tier review found one medium defect
  (deadline-overrun transient retried) fixed with a red/green regression and
  two wording/limit notes; dispositions independently verified. Synthetic
  and loopback tests only; no provider inference. `uncomment` unavailable,
  so the comment-hygiene pass was skipped.
- 2026-11-28T14:33:39Z: verification pass
- Final gate on the delivered code: Ruff, mypy, pytest 4059 passed,
  7 skipped (2787 s); Taskrail verify run 2026-11-28T14:33:39Z passed.
