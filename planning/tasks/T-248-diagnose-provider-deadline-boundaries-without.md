---
id: T-248-diagnose-provider-deadline-boundaries-without
title: Diagnose provider deadline boundaries without retries
status: todo
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-247-enforce-explicit-per-card-exercise-and-nested
updated_at: "2026-11-27T16:40:12Z"
---

# T-248-diagnose-provider-deadline-boundaries-without Diagnose provider deadline boundaries without retries

## Description

F2 of the approved post-trial package. After accepted F1, diagnose actual deadline
boundaries without assuming remote latency is the cause. Own existing transport
timing/report boundaries sequentially with other provider writers. See
docs/v0.5.0-post-trial-plan.md.

## Acceptance

- Audit retained elapsed time, completion usage and request-ID evidence through the
  private trial coordinator's bounded sanitized inventory. Distinguish local startup
  and postresponse overhead from known remote timing. Missing historical evidence
  stays unknown; current status markers follow body read and may not distinguish
  header/body delay.
- If needed, add only sanitized header/status/timing markers needed to distinguish
  wait-for-header and body delay. Preserve the credential-capable worker, screening
  before persistence, dropped-body behavior, cancellation and unknown accounting.
  Do not infer distinctions the evidence cannot support.
- A controlled local delayed response and near-deadline complete response exercise
  whichever distinctions are claimed, retaining reservations and no duplicate dispatch.
  Reuse existing sufficient HTTP-error coverage. A diagnosis that historical root
  cause cannot be recovered is legitimate.
- Do not increase the 60-second ceiling, add retries/backoff/jitter, make hosted
  timing probes, switch providers/models or dispatch unreserved attempts. The launcher
  defect is already fixed; do not create another production fix for it.

## Verification Notes

Follow implementing-changes, scoped checks, simplification and independent review/
dispositions. Executable changes require final Ruff → mypy → full pytest before
verify/complete. A documentation-only conclusion uses relevant structural checks
and explicitly records that no product change was needed. Commit/push main after
local gates without waiting for CI. Parent owns successors and final CI supervision.

## Implementation Notes
