---
id: T-231-qualify-minimax-provider-accounting-and-bounded
title: Qualify MiniMax provider accounting and bounded execution
status: completed
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-189-add-explicit-providers-and-blind-linguistic
updated_at: "2026-11-20T16:57:29Z"
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

## Current P2 acceptance and authority disposition — 2026-11-20

The maintainer's revision 55 execution approval now governs P2. T232/P1 is
completed on remote main at 7f4be5ed0f957514c3429b38810cf8dde0ad9690;
the coordinator reports exact-head Build 44768222639 and CodeQL 44768222549
successful. This is the actual implementation prerequisite. Installed Taskrail
offers no dependency-edit command; retain original dependency history and record
this prerequisite here rather than hand-editing lifecycle metadata.

The following mode-specific acceptance supersedes the original guaranteed-bound
requirement for this estimated-mode outcome only. All earlier requirements,
blockers, failures and two-probe evidence remain historical, not retroactively
passed. Guaranteed accounting stays default and token_bound keeps its meaning.

- Qualify only HTTPS https://api.minimax.io/v1/chat/completions with
  MiniMax-M3.1-Flash in the installed same pipeline under explicit estimated
  accounting. Bind application admission to reviewed endpoint/model/wire/settings/
  caps, not a user-authored qualification assertion; preserve unpinnable deployment
  revision and pricing/billing uncertainty. No fallback or linguistic authority.
- Initially spend at most 12 real dispatched attempts, including unknowns/retries,
  solely from the new revision55 shared DEVELOPMENT allocation. Retain one durable
  central campaign across handoffs: 2000 attempts, 40000000 estimated tokens,
  USD25 estimated public-rate valuation, 86400 reserved execution seconds;
  development/study/native/QA phase caps 80/1600/160/160. Existing closed probe
  spending/slots do not transfer. More than 12 requires coordinator allocation.
- Every complete request is at most 16 KiB, reserves at most 2048 output tokens
  including reasoning, has at most 60-second timeout, and runs alone. Preserve
  per-run 200 attempts/200000 estimated tokens, prospective reservations, all
  outcomes, usage/pricing review stops, no automatic retry/unknown resubmission,
  full reservation retention and actual reported usage separate from estimates.
- Check applicable current public pricing before dispatch. Existing environment
  key may consume existing Credits/balance; no top-ups/account changes or claim of
  a guaranteed financial ceiling. Use only newly authored manifested synthetic
  Latin/German development data; no private/protected/reference/holdout/judge data.
- Exercise installed proposal and genuinely proposal-free counter-check with
  strict task binding and complete outputs; retain bad semantic answers. Exercise
  replay and malformed/length/tool/HTTP/unknown/cancellation controls offline or
  loopback. Development custody cannot produce learning exports. Qualify runtime
  wire equivalence offline without borrowing native/QA attempts or calling a
  runtime-purpose hosted request in this development phase.
- Use red/green TDD, dedicated simplification, independent risk-scaled review and
  verified dispositions, then an independent fixed-candidate installed/public-
  boundary checkpoint without reviewer hosted calls. Preserve prior failures and
  review budgets. Freeze bytes and pass exact Ruff -> mypy -> pytest final chain
  plus Taskrail/guards before supported verify/complete. Genuinely completed P2
  may commit/push to main with hooks/maintainer identity/no attribution trailers,
  remote read-back, exact-head subscription and Build/CodeQL success.

No automatic policy enablement, T200/T201 lifecycle, references, judges, studies,
publication/deployment or successor selection belongs to this task. Transfer
credential-free exact campaign commitments and remaining global/phase spending
for continuation; no later owner may reset the campaign allocation.
- 2026-11-20T16:23:28Z: Revision55 mode-specific P2 acceptance adopted after completed T232 prerequisite; historical guaranteed-bound outcome retained

## Current P2 execution evidence — 2026-11-20

Reconciled own stale task/STATE drafts with accepted remote P1 before writing;
published T231 exactly matched the preserved dd571c82 task commitment. Supported
unblock/validate/next selected this exact task with off_spec=false before start.
No other owner, remote drift, task recreation or successor selection was observed.

One new central revision55 campaign is authoritative across all later runs:
5bb4efba89e50a95713e9b9bda958adab8a1b5e9c90f8e74f57e6e2a202c5d7f.
It retains four development attempts (two failed, two structurally succeeded),
71314 estimated input-plus-output tokens, USD0.014384 public-rate reservations,
240 reserved execution seconds and 42.588 observed execution seconds. Actual
provider-reported usage is 4390 input + 3374 completion = 7764 tokens; completion
includes 2447 reasoning tokens. None of these observations is an invoice or bound.
Remaining global allocation: 1996 attempts, 39928686 estimated tokens,
USD24.985616 valuation and 86160 reserved seconds. Remaining phase attempts:
development76, study1600, native160, QA160. Four of the initial12 P2 attempts used;
no further calls are planned. Old two-probe allocation stays closed and separate.

The first new hosted proposal exhausted default thinking at length/empty content;
the second, with low effort, returned stop but violated the existing selected_sense
validator and falsely called silva masculine. The final installed wire uses
max_completion_tokens2048, low effort and MiniMax structured-answer contract/1
explaining that existing validator without weakening it. Exact same proposal-free
task was sent separately to proposal and counter-check; both parsed fully. Proposal
gave silvas/silvās; counter falsely gave silvae as accusative plural. All failures
and wrong answers remain evidence, never linguistic acceptance or reference truth.
Development export was not requested because custody forbids learning exports.

Final successful wire sizes were 3884/3881 bytes. Candidate providers.py SHA256:
c46519a61d175e12a94b18c57685be746b63c55c7f59803f887be8b1f0dbc412.
Central ledger SHA256 after all four attempts:
0584049811c31cef471d7a0513aa1b24188e5b3d586cb2b1e17b91d1154b147c.
Credential-free receipts, scope/permission/request commitments, original central
state and continuation manifest are retained for transfer; never initialize a
replacement campaign or restore only a run. Public pricing was checked live before
calls: USD0.15/M input and USD0.60/M output. A curl snapshot initially retained an
empty redirect body; following the public redirect saved the verified rates. No
account-side deduction/tier entitlement or financial cap was inferred or changed.

Red admission and structural-instruction regressions failed for their intended
reasons, then passed. Dedicated simplification made no changes and provider tests
passed103; mypy passed239 source files. A zero-call unsupported second-declension
preflight and a post-persistence display KeyError are retained separately from
hosted outcomes. No inference was retried because of that display error. Independent
review/checkpoint, frozen full final gate and delivery remain pending at this note.
Evidence: https://ampcode.com/artifacts/t231-minimax-qualification-af346b0f-6810-4e7d-9976-32ca2699da23.

Subsequent review/checkpoint disposition: high-risk conventional review completed
general, security/data-integrity, Python/tests and separate claims/comment-audit
lanes, plus one shared boundary/overall adversarial round. No confirmed defects
remain. C-01 (configured60 versus permission30) was independently rejected as a
defect: adopted acceptance is at most60, documentation qualifies configured60,
the wire has no timeout field and tighter consent must remain effective. No
correction/follow-up round or budget reset occurred.

Independent installed checkpoint passed103 provider checks with the key unset and
external networking impossible. All123 packaged files matched source/wheel/install;
positive runtime admission was offline only. Four actual captures replayed with
original outcomes; all three copied runs resumed offline safely, then direct
learning export refused with no CSV and unchanged copied ledgers. Failed build/
harness iterations remain retained. Wheel SHA256:
647303406aa4ab9c47a6af9b9b8bac36c5e85017d4d397e3660b97daae542471.
No reviewer made hosted calls. No mutation adapter was installed or set up.

Credential-free campaign continuation was uploaded through supported transfer to
the coordinator, 131007 bytes/203 verified payload hashes, archive SHA256
5d6ceeadf60d3adcd3a250485084f4b2da8dfc23b18bc2d30bec198e30685bc1.
It preserves exact central state and all charged runs; restoration must compare
existing state and cannot initialize empty spending or overwrite a newer ledger.
The owner retains this checkout; final frozen full gate and delivery still remain.

Final frozen-code gate completed at 2026-11-20 16:56:53 UTC: uv run ruff check
passed, uv run mypy passed239 source files, then uv run pytest -v passed3883,
skipped7 with one warning in999.71seconds. Executable diff SHA256 remained
bb9ece98e1428c93c238f31a8efe4fc63ad3fd09f67d1290f845c2ac1c535598;
provider bytes still match the reviewed and installed-checkpoint candidate above.
Taskrail validation and whitespace checks passed; fetched origin/main remained
the accepted P1 base. This completes the amended estimated-mode protocol scope,
not guaranteed accounting, linguistic reliability, acceptance policy or publication.
Direct-main delivery and exact-head Build/CodeQL acceptance remain separate gates.
- 2026-11-20T16:57:29Z: verification pass
- 2026-11-20T16:57:29Z: Completed amended estimated-mode scope after frozen full gate at 2026-11-20 16:56:53 UTC, independent review and installed checkpoint. Direct-main delivery and exact-head CI are supervised separately; automatic acceptance/export and guaranteed-bound claims remain disabled.
