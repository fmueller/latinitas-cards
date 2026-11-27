# Optional provider boundary (T-189)

`latinitas_cards.providers` is the provider-only API for later pipeline integration.
It does not register a CLI command, select source material, render cards, create
acceptance decisions or enable policies. Production still ships only
`v050-disabled`, revision `1`. T-190 owns semantic acceptance and T-193 owns
orchestration. Existing offline commands do not import or invoke this API.

## Configuration and disclosure

Construct `ProviderRun` with independent explicit `proposal` and `counter_check`
`Endpoint` values (either may be absent), the allocated run ID, effective config
digest, owner-only run directory and `Limits`. No endpoint/model is inferred or
substituted. `openai` means a non-streaming OpenAI-compatible chat-completions
endpoint; `ollama` means a non-streaming Ollama chat endpoint. Supply the full URL,
including its path. `latinitas-json/1` is the direct strict task/answer protocol used
by controlled test servers, not a claim of real-provider support or reliability.

### Output capabilities

Select `provider: openai-json-schema` explicitly for a Chat Completions endpoint
and model documented to support strict Structured Outputs. For example, OpenAI's
`https://api.openai.com/v1/chat/completions` with `gpt-4o-mini` or
`gpt-4o-2024-08-06` supports this mode. It sends `response_format.type: json_schema`
with `strict: true`, required fields and closed objects. Constants become singleton
enums; string patterns/minimum lengths and minimum array lengths remain local
checks, so the wire schema also fits the documented fine-tuned-model subset.
The full local answer schema remains in the prompt. No model is selected or changed
for you. Check the exact model's endpoint, token-cap and schema support before use.

For other compatible services, this selector is an explicit operator declaration
of documented strict-schema capability, not capability detection or certification.
Provider errors and invalid answers fail without falling back to JSON mode or
another model. A third-party service may silently ignore unsupported parameters;
a locally valid answer does not prove provider-side schema enforcement.
Existing `openai` configurations retain `json_object` JSON mode, which does not
enforce a schema. Compatibility or accepting unknown parameters proves neither
JSON mode nor schema enforcement on a third-party service.

MiniMax Chat requests to `api.minimax.io` and `api.minimaxi.com` omit
`response_format`, and cannot select `openai-json-schema`. The official M3.1-Flash/M3
Chat request schema does not document that field. The separate native
`/v1/text/chatcompletion_v2` documentation limits `json_schema` to Text-01;
it does not establish support for M3 Chat. MiniMax therefore receives a prompt-only
answer schema, with no provider-side JSON/schema guarantee claimed.

Official sources rechecked 2026-11-27:
- https://platform.openai.com/docs/guides/structured-outputs
- https://platform.minimax.io/docs/api-reference/text-chat-openai
- https://platform.minimax.io/docs/api-reference/text-post

All modes retain strict local parsing, task binding and semantic validation.
Refusals, truncation, malformed JSON and extra answer keys fail closed, even when
the response claims to follow the schema. Valid structure does not establish Latin
correctness or whole-card acceptance. Capability selection, including the complete
wire schema, is committed in disclosure, request identity and pre-dispatch budgeting.
Existing endpoint record keys and historical receipt formats are unchanged. Changed
wire bytes need fresh permission and do not reuse an older completed invocation;
old ledger calls remain charged and readable. Live qualification is separate.

The orchestration caller must show `disclosure(task, scope, recipe=...)` before confirmation
or network use, including provider retention policy information. It contains both
effective endpoints/models, the complete outbound task, source fields/spans/language,
input commitments and call/token/time limits. Locality is explicitly declared, not
inferred from a URL. The whole path, including extraction and upstream services,
must be verified local before making a local-only claim. An orb loopback service
is **not the user's computer**. Unknown upstream locality is not local-only.

Resolve `credential_ref` only at runtime and pass the resolved secret to
`HTTPTransport(credential=...)`. Never put credentials in a URL, task, model name,
profile, source evidence or disclosure. URL userinfo, queries and fragments are
rejected. The transport does not log headers or provider exception text, consult
proxy environment variables, follow redirects, expose tools or execute model text.
Run artifacts are local owner-readable records; this says nothing about remote
provider retention.

Hosted proposal/check calls, including tasks extracting lecture notes, require a
strict `HostedPermission` matching the run, config, endpoint, expiry, source hash,
exact disclosed task and complete wire-request hashes and nonempty scope. Include
all hashes from disclosure's `input_sha256s` and `request_sha256s` in the permission's
`input_sha256s`. Complete request commitments prevent changed prompt/configuration
data from hiding behind an unchanged field/span manifest. Requested fields and
Unicode codepoint spans must also be subsets of the approved scope. The trusted
source-selection caller owns the original field/span-to-text mapping; model/source
text cannot construct permission, endpoint or destination objects. Confirmation is
an explicit operator action, not model output. The eventual CLI owns confirmation,
permission persistence and displaying effective permission-capped limits.

## Blind requests and responses

`request` requires trusted recipe context and checks exact target keys against the
installed support registry, with feature types from the existing versioned feature
model. Unexpected proposal-answer/rationale/confidence target fields are rejected;
the caller still owns constructing objectives/evidence from original sources, not
from proposal results. Both roles use separate requests. Only installed role-specific
prompt revisions are allowed, not caller-provided prompt text. OpenAI/Ollama requests
include the output schema; the direct test protocol carries the same answer contract as
its `instruction`. Each request states that contract once, never inside task data.
No tools are exposed. The journal retains the actual credential-free HTTP request bytes and a
data-scope artifact, not just a digest. Model output must match the strict versioned
answer schema and task digest. Duplicate keys, nonfinite JSON, truncation, tool
calls and unexpected answer keys fail closed. Since `proposal/2` and `blind/2`, the
documented nested assertion values are read with the same strict JSON reader: form
`source`/`target`/`stem`/`rendered` objects and controlled-phrase sense mappings.
Malformed JSON, wrong shapes and control characters other than tab and line breaks
inside form values are `malformed_output`. A
well-formed form answer that ignores its explicit direction, prompt template, stem
line, literal stem concatenation or target surface is a separate `contract_violation`
failure. Both are retained, charged failures without repair or retry; neither
checks Latin correctness or explanatory prose. Credential-bearing reflected responses
(including supported JSON decoding layers) are rejected before persistence. Other raw responses are retained separately;
they are untrusted evidence, not instructions or acceptance records.

`compare_answers` preserves top-level and per-assertion alternatives and objections.
It recognizes only complete structural identity without open alternatives/objections;
other readings remain unresolved, not contradictions inferred from string inequality.
It is a payload comparison, not the class-specific semantic comparison check from
the approved contract. Even its `equivalent` result has
`authorizes_acceptance=False`. T-190 must bind the actual request/response digests,
all required class-specific evidence and semantic comparisons before evaluating
acceptance. Legitimate alternative wording or segmentation must not be discarded
merely because this structural comparison cannot resolve it.

## Opt-in estimated hosted accounting

Guaranteed accounting remains the version-1 default. `token_bound` still means a
trusted, proven upper bound over the complete service input; missing qualification
still refuses new requests. No heuristic is supplied through that callback.

Version-2 pipeline profiles explicitly add `processing.accounting` and set
`processing.retries` to `0`. This uses the **same** `pipeline run`, `stage` and
`resume` path, without a fallback provider or a second development runner. The
accounting record has format `latinitas-estimated-accounting`, version `1`, kind
`estimated`, a campaign digest, phase (`development`, `study`, `native`, `qa`),
purpose (`development`, `isolated_evaluation`, `runtime`), estimator
`wire-padded/1`, endpoint-specific public-pricing observations, `service_revision`
(a string or explicit `null` when unpinnable), and
`acknowledges_no_guaranteed_cap: true`. Its exact schema is exposed by
`EstimatedAccounting.model_json_schema()`. Version-1 profiles cannot contain this
record; version-2 profiles cannot omit it. Existing offline workflows are unchanged.

`wire-padded/1` reserves 2 × complete serialized UTF-8 request bytes + 8192 estimated
input tokens, plus the full requested output allowance. This is a padded heuristic,
**not a guaranteed service-token or financial ceiling**. Reasoning shares the
selected output cap (2048 by default); qualifying a real adapter's compliance is separate work.
Public rates are operator-checked observations bound to the exact endpoint/model
digest, with a public HTTPS source, check time and expiry. Rates are positive integer
microdollars per million input/output tokens. Each reservation rounds valuation
up to a whole microdollar, assuming no cache discounts. Missing, stale or mismatched
pricing blocks new spending. Reported usage and actual plan deduction are not the
same thing: deduction remains unknown unless separately evidenced. Existing
Credits/balance may be used; no account changes, top-ups or subscriptions occur.

Initialize an explicitly approved `latinitas-campaign-allocation` version-1 file
with `pipeline campaign --allocation FILE --initialize`. Inspection without
`--initialize` does not create an allocation. The allocation binds a campaign
digest and authority, maximum attempts, estimated tokens, microdollars, provider
execution seconds and all four phase caps. Ceiling values may be narrowed, never
increased above 2000 attempts, 40000000 tokens, USD 25 estimated valuation and
86400 seconds. Phase attempt caps are development 80, study 1600, native 160 and
QA 160; unused capacity cannot be reassigned. Reinitializing the same allocation
is idempotent, not a reset; a different allocation under that identity is refused.
Creating another allocation requires separate prospective authority, not a retry.
T231's closed two-call probe allocation is not imported or credited.

The authoritative ledger is under
`~/.local/state/latinitas-cards/hosted-campaigns/`, **outside run directories**.
Retain this owner-only state when restoring or moving the execution environment;
restoring only a run is not sufficient. Local filesystem administrators and
operator-authored permission/allocation files remain trusted; this is not a
tamper-proof remote billing service. A single application dispatch lock spans
campaigns and is held through response persistence. Runs and roles share campaign
spending, and each run is additionally limited to 200 attempts and 200000 estimated
tokens. Request wire bytes are limited to 16384 and deadline to 60 seconds.
Output defaults to 2048 tokens including reasoning. The full deadline is reserved before every
attempt; actual elapsed milliseconds are recorded separately, without refunds.

For development tasks, profiles may explicitly set `processing.max_output_tokens`
to an integer from 1 through 8192, for example `"max_output_tokens": 8192`.
Omitting it keeps the 2048 default and historical profile serialization; explicitly
selecting 2048 has the same identity. Both profile versions support the setting.
It reaches each adapter's exact output-cap field and appears in preflight disclosures
and effective report limits. Changed caps change request/stage identity and require
fresh exact-wire hosted permission (also a new effective-limits digest in version 2).
They cannot reuse a differently capped answer. Historical receipts, responses and
charges remain intact; the full new allowance is reserved before dispatch, not
refunded from reported usage. Larger answers leave less budget for other calls:
run, permission, phase, campaign, token, cost and time ceilings do not increase.

This is development capacity, not expanded runtime qualification. Estimated
`purpose: runtime` still requires the exact qualified 2048-token configuration.
Offline public controls use an independently interpreted complete phrase answer
and a synthetic byte-pair token counter; they do not establish hosted tokenization,
success, or linguistic quality. Every requested assertion, including full source,
anchor/role bindings and rendered prompt/answer claims, remains required. Use the
task's existing presentation contract concisely; do not abbreviate source identity,
omit claims, or repair malformed responses to fit. Truncation and invalid schemas
remain failures. More capacity does not fix wrong senses, recipes, segmentation
or unsupported claims; agreement still grants no acceptance authority.

Hosted estimated requests require permission version `2`, including all version-1
scope/request commitments, the exact accounting record, effective `Limits` digest,
permission microdollar/execution-second ceilings and
`possible_existing_credits_use: true`. Permission binds purpose and estimator, not
just endpoint and model. Renewing it cannot restore run, phase or campaign budget.
Every dispatch takes the strictest remaining permission/run/phase/campaign limit.
No hidden API retry occurs. Exact request/scope commitments and full reservations
are durable before sending. Raw request/scope artifacts and fully received,
screened responses within the transport capture limit stay in the owner-only run
store; other outcomes retain available identity/error/unknown evidence.
Version-2 reports identify estimated accounting, while
the versioned campaign provenance retains each reservation and provider-reported
input/output/total usage separately. Neither a low usage report nor cancellation
refunds any reservation.

Missing/invalid usage, or usage above either input or output reservation, stops
new campaign dispatch for review. A failed/unknown dispatch retains its full charge.
Unknown transport error labels map conservatively to `unknown`; raw exception
messages are not persisted. Unknown calls block the campaign until identified
reconciliation or explicit abandonment. Existing `pipeline reconcile` controls
apply; abandoning does not prove that remote processing or billing stopped.
For a crash after central preparation but before a run journal entry,
`pipeline campaign --allocation FILE --abandon-call ID --confirm-unknown-billing`
closes the orphan without refund. A diverged/stale run remains refused; use an
explicitly authorized new run with the same campaign and acknowledge potential
duplicate processing. Every potential duplicate requires both retry and duplicate
billing acknowledgment and fresh remaining budget. Review stops are not cleared by
abandonment. No automatic reset/replenishment or financial guarantee is implied.

Model labels and declared service revisions are retained. An unpinnable deployment
is a reproducibility limitation, not a fabricated immutable revision. A detected
material revision change refuses further spending/reuse and requires requalification.
Accounting cannot detect an undisclosed provider-side deployment change.

Accounting grants no linguistic, policy, acceptance or export authority. Purpose
and accounting configuration are immutable for a run. Development and isolated
evaluation captures are not reference truth or ordinary learning exports; direct
export refuses them and orchestration reports export as not requested.
The installed automatic registry remains disabled. Accounting does not perform
protected studies, reference solving or publication.

### MiniMax estimated-mode protocol qualification

The installed application admits estimated runtime dispatch only for this exact
declared scope; a profile or permission cannot add another qualification:

- Hosted `openai` protocol at `https://api.minimax.io/v1/chat/completions`, model
  `MiniMax-M3.1-Flash`, with explicit `service_revision: null` (unpinnable alias).
- The installed proposal/blind messages plus MiniMax structured-answer contract/1,
  which states the existing selected-sense validator invariant. The 2026-11-20
  observation below used proposal/1 and blind/1. The admission check does not compare
  prompt revisions; proposal/2 and blind/2 change request bytes and need fresh exact
  permission. This does not decide whether the changed prompts need requalification.
  Both roles receive the same source task, without the proposal answer.
- Nonstreaming text-only requests with a prompt-only JSON answer contract,
  `max_completion_tokens: 2048` and
  `reasoning_effort: low`, 60-second configured deadline and zero retries. No tools,
  multimodal data, alternate service tier, user-supplied system prompt or fallback.
- Unchanged strict stop/one-choice/task-bound answer parsing, version-2 hosted
  permission, applicable public pricing, estimated reservations and campaign guards.

Other endpoints/models/protocols/localities, a claimed pinned service revision or
different runtime output/deadline/retry configuration still refuse with
`estimated_runtime_not_qualified`. Guaranteed mode still has no installed input
bound. Development and runtime use the same serializer and admission does not
change artifact custody, the policy registry or learning-export eligibility.

Qualification records a bounded 2026-11-20 synthetic development observation,
not provider-side strict-schema enforcement, a tokenizer bound or Latin reliability.
Four new-campaign attempts were retained: default thinking exhausted its cap with
no answer; low effort returned schema-invalid selected senses and a false masculine
explanation; the explicit structural contract then produced two complete answers.
The proposal gave the useful accusative plural `silvas`/`silvās`, but the blind
counter-check falsely gave `silvae`. Structural success does not accept that claim;
neither agreement nor this qualification supplies linguistic or whole-card authority.
No runtime-purpose hosted request was used to qualify that historical wire.
The current serializer omits its undocumented `response_format: json_object`;
the historical observation is not live qualification of these changed bytes.
Fresh wire-bound permission is required; no hosted rerun is authorized by this change.

Observed public rates were USD 0.15/M input and USD 0.60/M output at
https://platform.minimax.io/docs/guides/pricing-paygo.md. Actual plan deduction,
Credits/balance allocation and invoices were not established. Provider-reported
usage stayed within each reservation in these samples; this is not a guarantee.
The service alias may change without notice; detected material changes require
requalification, and undisclosed provider-side changes cannot be detected locally.

## Guaranteed spending, cancellation and recovery

Defaults are 60 seconds/request, one retry, 200 total attempts, 200000 total
input-plus-reserved-output tokens, at most 2048 output tokens per attempt, and one
in-flight call under the run lock. Hosted permission caps configured limits by
minimum. Both roles and every retry consume the same durable ledger. Copying a run
with a different run ID/config fails; permission renewal does not reset spending.
Explicit higher total ceilings require corresponding confirmed hosted permission.

Each dispatch requires a trusted adapter-supplied **proven upper bound** for all
input tokens, including protocol/prompt overhead. Unknown tokenization rejects the
request. There is deliberately no byte/4 estimate or production tokenizer claim;
tests use a synthetic server with a controlled byte-token budget. Qualification of
actual supported endpoint revisions and token bounds remains gated in T-201.

Request and scope artifacts, the prepared reservation and the sent journal state
are durably written before dispatch. Responses and digests precede succeeded state.
The current API conservatively charges full reservations, even after success; it
does not trust model-supplied token usage to release spending. Known transient
failures alone may retry, after one second and with fresh budget. The per-request
retry ceiling survives reopen. HTTP errors, timeouts, disconnects and cancellation
after dispatch are unknown, not proof that inference was never executed. A
disposable HTTP I/O process bounds DNS/connect/TLS/body waits and is terminated on
cancellation/deadline. This cannot cancel processing or billing at the remote server.

Unabandoned unresolved dispatches block new work. Abandonment permits different
work while preserving the unresolved call and its charged reservation. Resume reuses integrity-checked
successes, does not repair malformed answers and never resubmits unknown calls.
`reconcile` can retain an operator-retrieved response only when the captured
provider request ID, original endpoint/request commitments and exact original task
match. Retrieval authentication belongs to the trusted operator/adapter; response
content cannot attest its own provenance. Without a captured identity or available
retrieval, `abandon` leaves the call unresolved and its reservation charged. Retrying
that abandoned request requires both `retry_unknown=True` and
`duplicate_warning_confirmed=True`, explicitly acknowledging possible duplicate
processing/billing, plus fresh remaining budget and current hosted permission.
Abandonment is not cancellation or acceptance.

## Evidence limits

The integration suite uses synthetic public fixtures, fake transports and controlled
local HTTP servers only. It exercises request capture, permission denial, malformed
output, objections, durable spending, unknown-call recovery, HTTP timeout/cancellation
and offline no-network commands. It neither runs a model nor establishes linguistic
reliability. No reference-judge study, private material acquisition, hosted inference,
model download, policy enablement or native Anki acceptance is performed here.
