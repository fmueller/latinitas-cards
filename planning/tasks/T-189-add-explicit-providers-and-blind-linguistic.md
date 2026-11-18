---
id: T-189-add-explicit-providers-and-blind-linguistic
title: Add explicit providers and blind linguistic counter-checks
status: completed
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-187-implement-versioned-automatic-decisions-and-reuse
updated_at: "2026-11-18T06:14:51Z"
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

- 2026-11-18T06:14:06Z: verification pass
- Added the provider-only API and caller contract in `docs/provider-boundary.md`.
  Proposal/check endpoints are independent; hosted permissions commit the exact
  disclosed requests and scope. Installed role prompts and target validation keep
  blind inputs separate. Structural agreement never grants acceptance authority.
- Preserved approved operational defaults, shared durable reservations, unknown-call
  reconciliation/abandonment, explicit duplicate-processing acknowledgment, bounded
  HTTP cancellation and credential screening. Production remains `v050-disabled`,
  revision 1; existing offline commands and T187/T188 authority are unchanged.
- TDD covered initial missing-module and durable retry-ceiling failures. Review
  regressions produced 10 failures before fixes; all 45 provider cases then passed.
  Controlled public HTTP exercised three protocols without model calls. Offline
  validate/preview/generate passed with sockets and provider transport forbidden.
  Crash-window and contention probes preserved charged unknowns and one dispatch.
- Dedicated simplification and independent high-risk conventional review completed.
  All eight verified defects and two wording issues were resolved in fix round 1/2.
  The independent overall challenge returned HOLDS; challenge budget used 4/5.
- Final exact gate at the verification timestamp above: `uv run ruff check` passed;
  `uv run mypy` passed for 214 source files; `uv run pytest -v` passed with 3494
  passed, 7 skipped and one existing Pydantic serializer warning in 540.28 seconds.
  Skips cover derived lock spellings and hardlinks without an existing inode.
- Reviewed executable SHA-256: providers module
  `205dcb1ca723b9b54be7ee1353d236345e61974ec123c2c2bcde9bd5a4cfc4fd`;
  provider integration tests
  `0669291cc5be9f23a18e6bd0a944ef7afb6d0756979e08c039e5e8f353fbb6d5`.
- No live LLM/reference judge, private acquisition, hosted inference, model download,
  policy enablement or native acceptance ran. Real endpoint/token qualification and
  full native acceptance remain gated in T201. Semantic acceptance and orchestration
  remain T190/T193 work. Task-local completion is not integrated/release acceptance.
- 2026-11-18T06:14:51Z: Provider-only plumbing implemented and reviewed; verification 2026-11-18T06:14:06Z passed. Integrated/native endpoint qualification remains gated in T201; outer loop owns acceptance.
