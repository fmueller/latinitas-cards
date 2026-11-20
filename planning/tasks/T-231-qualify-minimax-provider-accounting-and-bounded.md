---
id: T-231-qualify-minimax-provider-accounting-and-bounded
title: Qualify MiniMax provider accounting and bounded execution
status: blocked
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-189-add-explicit-providers-and-blind-linguistic
updated_at: "2026-11-20T12:22:07Z"
---

# T-231-qualify-minimax-provider-accounting-and-bounded Qualify MiniMax provider accounting and bounded execution

## Description

Qualify a concrete MiniMax adapter under the adopted provider-boundary contract.
This follows T-189 under specs/v0.5.0.md#cli-orchestration-and-provider-boundaries.
Provider-use approval covers only newly authored synthetic development data, not
reference acquisition, protected evaluation material or automatic acceptance.

## Acceptance

- Establish a source-backed complete-input token upper bound for the actual wire
  request and service framing, plus reserved output including thinking. Estimates,
  byte heuristics, matching samples and posthoc usage do not qualify this bound.
- Implement only a qualified MiniMax-specific route with compatible strict structured
  answers and task binding. Other endpoints retain missing-bound refusal; ordinary
  offline exact reuse remains zero-call. Preserve durable permission, cumulative
  spending, unknown reservations and replay; production acceptance stays disabled.
- Establish applicable model/account entitlement and billing limits before inference;
  do not assume subscription quota excludes purchased Credits. No account changes,
  upgrades, paid fallback or installed ceiling increases are authorized.
- When accounting, permission and billing prerequisites hold, execute a prospective
  synthetic checkpoint: at most four attempts including retries, 16 KiB newly authored
  Latin/German data and 2048 generated tokens per attempt, 60 seconds/request, one
  in flight, 15 minutes total, within the existing 200000-token cumulative ceiling.
  Retain identities, source commitments, actual request/response digests, outcomes
  and failures. Missing guarantees stop dispatch and block integration completion.
- Use red/green TDD for executable changes, dedicated simplification, independent
  risk-scaled review and bounded runnable-slice QA with positive/negative controls.
  Distinguish offline controls from hosted evidence; no artificial whole-card authority.

## Approved development-only exception

The maintainer's subsequent "yes, go ahead" authorizes a separate synthetic
compatibility checkpoint within this task, not completion of the qualification
acceptance above. For this checkpoint only, conservative ESTIMATED input-token
and cost accounting may replace a proven pre-dispatch bound. The 200000-token
cumulative ceiling is an estimated accounting limit here, not a guaranteed bound.
Existing Credits or account balance may be consumed with the existing environment
credential. No top-up, subscription or account-setting change is permitted.

The allocation is at most FOUR dispatched inference attempts total, including
retries, timeouts, unknown outcomes and malformed responses. Each complete request
is limited to 16 KiB, including newly authored synthetic Latin/German data and
instructions/schema, and reserves at most 2048 generated tokens including thinking.
Use a 60-second request deadline, one in flight, no automatic retries and a
15-minute probe window. Persist each reservation before dispatch; never release
unknown reservations or reset the allocation on resume. Retain exact request,
response/error and usage receipts with hashes and accounting uncertainty, but no
credentials. Stop when a usable checkpoint is established or the allocation ends.

This exception authorizes no production dispatch or fake bound, protected/private
data, references/holdouts, judges/evaluation, policy enablement, automatic acceptance
or export authority, T200/T201 lifecycle, release or publication. Production's
guaranteed-bound requirement and existing offline reuse remain unchanged. The
original blocker and failed evidence below are historical and remain valid for
production qualification. Probe success alone cannot complete this task's title
or original qualification acceptance; report its actual blocked/delivery state.

## Verification Notes

Run affected checks during development. Before verify/complete or delivery of code,
freeze executable bytes and run uv run ruff check, uv run mypy, uv run pytest -v in
that order, plus Taskrail validation and applicable guards. Record exact revisions,
review dispositions and any prerequisite blocker. A blocker report alone is not a
completed integration. Completed delivery requires exact-head Build/CodeQL success.

## Implementation Notes

- 2026-11-20T05:26:24Z: MiniMax complete-service-input upper-bound guarantee and account-specific billing cap are unestablished; estimate/count endpoints do not yet qualify actual wire/framing, so no hosted inference or integration completion
- Investigation at accepted main 5aea0f8b6f0b6717a9aae341e1a62c8075aac321:
  official Responses input_tokens documentation calls its result an estimate.
  Anthropic compatibility calls count_tokens both exact image usage and token
  estimation; no unambiguous complete text/system/schema/service-framing bound
  tied to generation was established. Moving both counting and generation to the
  same protocol would remove wire mismatch, not establish the missing guarantee.
- Independent contract challenge confirmed the unresolved prerequisite. A
  documented never-undercount promise or bounded correction could qualify; exact
  equality is unnecessary. An equivalent enforced admission cap could also qualify,
  but the documented 1000000-token context cannot fit the 200000 cumulative ceiling.
  No suitable caller-settable complete-input cap was established. This is a gap in
  qualification evidence, not a claim that MiniMax can never be supported.
- Official OpenAI/Anthropic documentation supports a 2048 generation cap including
  thinking. It may stop before any final answer. Current Chat request schema does
  not document response_format; runtime strict-output compatibility remains untested.
  M Plan usage rules consume purchased Credits after quota; exact key entitlement
  and an enforceable financial boundary remain unestablished.
- Sources: https://platform.minimax.io/docs/api-reference/responses-input-tokens.md,
  https://platform.minimax.io/docs/api-reference/text-anthropic-api.md,
  https://platform.minimax.io/docs/api-reference/text-chat-openai.md and
  https://platform.minimax.io/docs/m-plan/usage-rules.md. Source commitments and
  independent findings: https://ampcode.com/artifacts/t231-minimax-qualification-af346b0f-6810-4e7d-9976-32ca2699da23.
- Credential presence only was checked. This worker made zero authenticated
  metadata requests, token-count POSTs or inference attempts. No hosted request
  was prepared, so there are no hosted response digests or unresolved reservations.
  No protected material, fake production bound, cap increase or acceptance authority
  was introduced. Product source/tests and inactive v0.5.5/T-229 remain unchanged.
  No verify/complete, commit/push or release occurred; this is not a completed
  integration. Implementation TDD/simplification/code review and the final Python
  gate remain pending a supportable adapter, not claimed passed for this blocker.
- Independent bounded offline QA passed 31 final scenarios: 21 provider controls,
  seven orchestration controls and three scratch probes. Concrete MiniMax URL/model
  refused missing production bounds or permission with zero dispatch. Exact replay
  through orchestration reused synthetic successes after restoring the production
  None bound, with unchanged ledger and no transport construction. Complete-wire,
  output reservation, durable unknown/retry spending and disabled authority controls
  passed. These are offline engineering controls, not hosted qualification.
- Retained the initial scratch result of two passes and one fixture failure:
  permission timestamps used +00:00 rather than the required Z. Only the scratch
  timestamps were corrected; rerun passed all three. No product fix was needed.
  Final Taskrail validation and git diff --check passed; full Python gate was not
  run for planning-only blocked work. No executable tracked bytes changed.

## Development exception execution

- 2026-11-20T12:14:45Z: Maintainer approved four-attempt synthetic-only estimated-accounting/Credits exception; resume T231 for compatibility checkpoint only, original production qualification remains unresolved
- Fresh origin/main and HEAD still matched the accepted base; no competing active
  lifecycle owner or new hosted activity existed in this checkout. Recorded the
  exception above before supported unblock, authoritative selection with
  off_spec=false and start. Independent read-only probe preflight found no material
  safety blocker; no requests were sent by that reviewer.
- Two attempts were dispatched to https://api.minimax.io/v1/chat/completions with
  MiniMax-M3.1-Flash. Both returned HTTP 200, finish_reason=stop and strict complete
  ModelAnswer/task binding through the installed parser. The first used unchanged
  current serialization: max_tokens=2048, response_format=json_object, stream=false,
  default reasoning. It returned amās and a correct German second-person singular
  present indicative active description. Reported input/output/reasoning: 1180/738/496.
- The second used scratch-only max_completion_tokens=2048 and reasoning_effort=low,
  retaining JSON response format. It returned monēmus and first-person plural present
  indicative active, but added the false explanation "Genus verbi: erste Konjugation
  (moneō)". moneō is second conjugation; conjugation is not genus verbi. Strict shape
  validity does not establish correct linguistic content. This semantic failure is
  retained unchanged, not repaired or promoted to acceptance. Usage: 1197/429/175.
- Complete wire sizes were 4056 and 4096 bytes, below the stricter 16 KiB total-wire
  limit. Request durations were 3.384 and 3.346 seconds; first dispatch to final
  response took 26.455 seconds. Allocation closed at 2026-11-20T12:19:24Z with two
  of four attempts used, no retries/timeouts/unknown outcomes and no more dispatch.
- Estimated input was twice complete UTF-8 wire bytes plus 8192, explicitly a
  heuristic NOT a proven bound. With 2048 output reserved per attempt, cumulative
  estimated reservations were 36784, below the estimated 200000 ceiling. Both full
  reservations remain charged. Actual reported totals were 2377 input and 1167
  generated tokens including 671 reasoning tokens, not a prospective guarantee.
  Public-rate conservative estimated spend was USD 0.0073608. Reported usage priced
  entirely uncached at USD 0.15/M input and USD 0.60/M output gives USD 0.00105675;
  neither figure is an observed account charge or financial ceiling. Existing
  Credits/balance consumption was approved; no account settings were changed.
- Exact credential-free request/response/answer and source commitments, durable
  allocation and raw usage/error receipts are retained with hashes in the linked
  evidence artifact. No reference/holdout/private data or production bound was used.
  A pre-dispatch scratch preparation failed because canonical JSON forbids float
  pricing; only scratch receipt encoding was corrected, before any attempt.
- The development checkpoint establishes limited real API compatibility and one
  useful synthetic answer, plus an observed linguistic failure. It does not satisfy
  the original provider qualification title/acceptance. Production accounting and
  dispatch remain unchanged/fail-closed; no automatic acceptance/export or T200/T201
  lifecycle is authorized. No verify/complete or completed-task delivery is claimed.
- 2026-11-20T12:22:07Z: Development-only exception checkpoint closed after 2/4 attempts: two complete strict answers, one linguistic failure retained; production complete-input bound still unqualified, so original integration acceptance cannot complete

## Subsequent amendment and execution authority

The maintainer subsequently approved and ordered execution of revision 55 of
https://ampcode.com/artifacts/v0-5-0-unblocking-plan-d8da3973-9b6a-451c-8ce1-93f5c4004857:
"Approve revision 55 and execute the MiniMax release-readiness plan under its
limits; stop before publication." This is subsequent authority, not a revision
of the historical two-probe outcome, its original qualification requirements or
its closed allocation. No unused probe slots transfer into the new plan.

The approved amendment permits explicitly opted-in estimated hosted accounting
for development, isolated evaluation and eventual qualified normal runtime in
the same supported pipeline. Guaranteed accounting remains the default and
token_bound retains its proven-bound meaning. Estimates are not guaranteed
service-token or monetary bounds and grant no linguistic or export authority.
The plan's dependent phases and separate freeze, enablement, genuine curated
approval, readiness and publication decisions remain as stated in revision 55.

P1 is materialized as T-232-implement-opt-in-estimated-hosted-accounting-with,
a high-priority T-189 follow-up under the existing provider-boundaries anchor.
T231 stays BLOCKED pending P1 and actual mode-specific runtime qualification
under P2; the probes did not satisfy it. Any qualification reconciliation must
explicitly adopt the new mode and record observed limitations, not relabel a
heuristic as a proven bound. This administrative handoff starts no P1 work,
authorizes no new hosted call, changes no runtime and completes no task.
