---
id: T-256-bounded-model-retries
title: Add bounded two-retry recovery for transient model failures
status: todo
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-254-replay-approved-ten-german-cards-after-post-trial
updated_at: "2026-11-28T13:23:46Z"
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
