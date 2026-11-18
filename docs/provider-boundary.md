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
include the output schema; the direct test protocol uses its versioned answer contract
out of band. No tools are exposed. The journal retains the actual credential-free HTTP request bytes and a
data-scope artifact, not just a digest. Model output must match the strict versioned
answer schema and task digest. Duplicate keys, nonfinite JSON, truncation, tool
calls and unexpected answer keys fail closed. Credential-bearing reflected responses
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

## Spending, cancellation and recovery

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
