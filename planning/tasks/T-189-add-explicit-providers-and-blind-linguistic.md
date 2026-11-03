---
id: T-189-add-explicit-providers-and-blind-linguistic
title: Add explicit providers and blind linguistic counter-checks
status: todo
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-187-implement-versioned-automatic-decisions-and-reuse
updated_at: "2026-11-02T23:10:11Z"
---

# T-189-add-explicit-providers-and-blind-linguistic Add explicit providers and blind linguistic counter-checks

## Description

P50-2 optional transport adapters and blind counter-check stage, also covering Shared
Evidence And Automatic Acceptance. Own payload validation, not rendering or orchestration
policy. Follow approved contracts and spec execution/validation requirements.

## Acceptance

- Configure proposal/check local or hosted endpoints independently, with the same
  or separate models. No mandatory hosted service or silent fallback. Display effective
  endpoints, sent data and limits; require explicit hosted permission before requests,
  including lecture-note extraction. Never log credentials.
- Capture actual counter-check input: source evidence/task present, proposal answer,
  rationale and confidence absent. Compare structured results under approved alternate
  reading/objection semantics. Agreement alone cannot establish acceptance.
- Treat source/model text as untrusted data with no tool/file/destination authority.
  Adversarial strings cannot expand request scope or permissions.
- Bound timeouts, retries/calls/tokens and cancellation. Malformed output, outage,
  exhausted limits and unknown calls remain failure/deferred, never accepted checks.
- Offline validation/render/export makes no calls. State whole-path locality and
  that an orb loopback endpoint is not the user's computer.

## Verification Notes

Inspect captured requests using controlled transports, permission/no-network controls,
malformed/disagreeing and matching-with-objection responses. Real authorized endpoints
belong to integrated acceptance; no private processing without permission. Run the
mandatory ruff/mypy/pytest chain.

## Implementation Notes
