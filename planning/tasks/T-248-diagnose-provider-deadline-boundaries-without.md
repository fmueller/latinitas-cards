---
id: T-248-diagnose-provider-deadline-boundaries-without
title: Diagnose provider deadline boundaries without retries
status: completed
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-247-enforce-explicit-per-card-exercise-and-nested
updated_at: "2026-11-27T23:26:05Z"
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

Base: accepted `b6fe8efffb3b7c731ae38d20061357eb59cd184d`. The private coordinator
supplied a sanitized inventory of all 17 retained development requests. It was not
committed and contains no card text, raw bytes, endpoint URLs or request IDs.

Diagnosis (sanitized metadata only):
- Four calls ended `unknown`/`deadline_exceeded` at 60005, 60008, 60011 and 60013 ms.
  None captured a body or a request ID. This matches the parent's own 60 s poll
  deadline plus the last poll wait and the time to stop the worker.
  The historical record cannot show whether startup, connect/TLS, header wait or
  body read consumed the time. Those causes remain unknown, and they are not
  attributed to provider latency.
- Twelve responses were captured, in 9998–54047 ms. Elapsed time did not track the
  reported completion or reasoning tokens. For example, 54047 ms produced 3719
  completion tokens, while 19211 ms produced 4008. Remote generation time is
  therefore not established.
- No retained request, successful ones included, recorded an `x-request-id`. Before
  this change, the worker reported status only after reading the body. The saved
  evidence therefore had no header-arrival boundary at all.
- Per the coordinator's inventory, the 483 ms early unknown is the old launcher
  failure, which is already fixed.
  It was not reopened.
- Some successful stage records lack call IDs. Their call roles were not inferred
  from ledger order.

Change: the disposable worker now signals when it starts and when response headers
arrive, together with the integer status. The status no longer waits for the body.
The parent records the receipt time of each signal and of the screened body. After
the ledger and charge are durable, each HTTP dispatch that returns or fails writes
sanitized `<call_id>.timing.json` milliseconds for those boundaries, plus the total
elapsed time including postprocessing. An interrupt during the transport call loses
the marks. The charge's
`elapsed_milliseconds` uses the same value. The 60 s ceiling, single dispatch,
lack of retries, screening, dropped bodies, process termination and
unknown/failure charges are unchanged.
The timing file contains only integer milliseconds or null under fixed keys. It holds no
status text, request ID, credential or response bytes, and fake transports write
none.

Evidence: red came first. The three new timing assertions failed because
timing.json was missing.
- Green: a controlled local server sends headers and then withholds the body past
  a 2 s deadline. The result is `unknown`/`deadline_exceeded`, with a header mark
  and a null body mark.
- A complete response at 1.5 s succeeds under the same deadline.
- Each case dispatches once and keeps one charge with the reserved execution
  seconds.
- The existing timeout, 503-error and incomplete-framing (dropped-body) tests now
  also assert timing.
- A new fault-injection test shows that a failed timing write leaves the call and
  charge final.
- The remaining HTTP-error and cancel coverage is reused.
- Tests added: three. Removed: none.

Review:
- The simplifier pass bound one fixture payload and found nothing further in the
  review fixes.
- The `uncomment` comment-hygiene pass was skipped because the tool is not
  installed.
- The first, sequential high-risk review found that non-200 responses dropped
  timing. That was fixed with a red/green regression.
- Independent read-only specialists covered general/Python, security, accounting,
  tests, claims and adversarial verification. They found that the timing write
  preceded durable accounting; it now follows `_save`, with a red/green test.
- The dropped-body timing path is now asserted and checked by mutation.
- Prose that overclaimed was corrected, and a flaky bound that caught nothing was
  removed.
- Dropping the near-deadline case was declined because Acceptance requires it.
- Optional cancel-after-reply timing coverage and the wall-clock margin of about
  500 ms were deferred.
- 2026-11-27T23:26:05Z: verification pass
