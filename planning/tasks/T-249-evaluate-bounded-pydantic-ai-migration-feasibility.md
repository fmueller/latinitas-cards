---
id: T-249-evaluate-bounded-pydantic-ai-migration-feasibility
title: Evaluate bounded Pydantic AI migration feasibility
status: completed
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-248-diagnose-provider-deadline-boundaries-without
updated_at: "2026-11-28T00:00:51Z"
---

# T-249-evaluate-bounded-pydantic-ai-migration-feasibility Evaluate bounded Pydantic AI migration feasibility

## Description

M1 of the approved package: bounded feasibility/design after accepted F1/F2, not
production adoption. Pydantic AI is preferred, not a demonstrated fix for observed
card failures. Deliver adopt, defer or infeasible and an independently reviewed exact
migration receipt. A1 and optional R1 always use fixed current transport before M2,
regardless of message equivalence. See docs/v0.5.0-post-trial-plan.md.

## Acceptance

- Use one disposable prototype with synthetic data and at most one targeted correction
  after a diagnosed failure; unresolved feasibility returns defer/infeasible. Evaluate
  exact pinned pydantic-ai-slim[openai] (reviewed candidate v2.76.0), Python 3.13/3.14
  and project constraints. Registry acquisition is permitted for disposable setup;
  runtime probes block external networking and use local endpoints. Do not mutate
  the production dependency or lockfile.
- Inventory resolved dependencies, licenses, runtime/install implications. Slim still
  includes substantial dependencies: httpx2/truststore, OpenAI/tiktoken, genai-prices,
  OpenTelemetry API, griffelib, anyio and pydantic-graph in reviewed upstream. Check
  exact tagged source, not drifting documentation alone.
- Find a supported API reducing hosted plumbing while preserving pre-normalization
  bounded bytes and the disposable-process wall-clock deadline. Reuse existing
  Record/ModelTask/ModelAnswer and validators, not competing schemas or a bespoke
  second framework. Keep one fresh per-card request per role, checker answer-free.
- Demonstrate exact before-run byte disclosure/permission, then capture dispatch.
  Either deterministic serialization equals dispatch or a reviewed transport hook
  constructs/discloses/obtains permission and reserves before sending. A post-send
  hash comparison is not consent. Record or strip X-Stainless headers and disclose
  added metadata; body identity alone does not authorize a wider data boundary.
- Enforce one HTTP send per logical call at the application boundary, independently
  of SDK settings. Local receivers observe one send on transient errors and invalid
  output. Disable transport/SDK/output retries (upstream SDK 2/output 1 defaults).
  Framework counters do not replace durable predispatch reservations.
- Preserve output cap and MiniMax reasoning settings; explicitly select native mode
  only for supported OpenAI and prompted mode otherwise. Assert no response_format
  on actual bare MiniMax-M3.1-Flash requests despite profile JSON-mode defaults.
  Verify the exact profile; no assumed MiniMax native enforcement. ToolOutput/tool
  acceptance is a separate unapproved contract change.
- Keep raw finish_reason=stop, refusal/tool rejection and strict duplicate/nonfinite/
  control/nested-value checks before normalization. Parseable cut-off text must fail.
  No repair, regeneration, fallback or coercion. Demonstrate process-enforced deadline,
  response bounds, no redirects/proxy expansion and credential-capable screening
  before unsafe persistence.
- Probe startup, schema/token counting, pricing/telemetry hooks, offline validation/
  export and a model run with external networking blocked. Unexercised lazy paths
  remain unknown. Compare ssl.create_default_context trust roots/environment CAs
  with httpx2/truststore/TLS verification. Preserve the trust boundary or stop for
  approval of a documented change.
- Deliver exact versions/output mapping, message/parameter/schema differences,
  SDK defaults, PromptedOutput templates and native _strict_schema projection,
  wire/proof/reservation design, qualification consequences, old-store read/resume
  strategy and rollback. Specify the smallest ownership boundary and reduced code,
  or why adoption is infeasible. A decision-needed contract change cannot be adopt.
- Independently review the receipt before M2 becomes executable. M1 completion does
  not authorize production dependency changes, hosted calls, tools, repair or policy
  enablement. Defer/infeasible still permits A1; never force a winner.

## Verification Notes

Use implementing-changes with proportionate prototype/design evidence, simplification
and independent conventional review. Separate executed dependency/Python/network
controls from unknowns. Structural/doc checks apply; retained executable changes
require final Ruff → mypy → full pytest before verify/complete. Commit the decision/
evidence summary on main without secrets or machine-specific artifacts; do not wait
for CI. Parent owns receipt acceptance and successor selection.

## Implementation Notes

Decision: defer, recorded in docs/v0.5.0-m1-pydantic-ai-receipt.md. Evaluated exact
pydantic-ai-slim[openai]==2.76.0 (PyPI; file-identical to tag v2.76.0, cdb587e) on
CPython 3.13.5 and 3.14.7, offline under a loopback-only network namespace with a
scrubbed environment and synthetic data. One disposable prototype, one targeted
correction; no hosted calls. The gated design keeps one send, strict raw validation,
the deadline and TLS trust only by delegating to HTTPTransport and _answer, so it
removes no plumbing, and the MiniMax wire change needs a maintainer-approved
contract change and T201 requalification. Product dependencies, uv.lock, dispatcher,
policies and store contracts are unchanged. Independent medium-tier conventional
review (general, evidence-fidelity and security lanes plus coordinator disposition
check) found no blocking issue; all candidates were fixed in the receipt or recorded
as unknowns, and extra probe runs were declined under the correction cap.
Documentation-only: git diff --check and taskrail validate; no Python suite applies.
- 2026-11-28T00:00:51Z: verification pass
