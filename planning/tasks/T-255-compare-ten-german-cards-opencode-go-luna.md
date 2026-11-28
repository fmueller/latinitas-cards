---
id: T-255-compare-ten-german-cards-opencode-go-luna
title: Compare ten German cards through OpenCode Go Luna
status: completed
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-254-replay-approved-ten-german-cards-after-post-trial
updated_at: "2026-11-28T15:15:07Z"
---

# T-255-compare-ten-german-cards-opencode-go-luna Compare ten German cards through OpenCode Go Luna

## Description

Run the maintainer-approved same-ten-card comparison through OpenCode Go using
GPT-6 Luna if supported and available, otherwise GPT-5.6 Luna. Compare against
the completed MiniMax replay without incorporating concurrent product fixes.

## Acceptance

- Pin executable c3a3523 / replay receipt 2b04621. Preserve the ten selected
  German-instruction Latin entries, objectives and strict application reader;
  exclude Stammformen, other decks, collection history and the full collection.
- Discover current Go availability and use the subscription's supported OpenCode
  path, not a direct API workaround. User approved the scoped comparison cost.
  Keep one fresh generation and one dependent answer-free blind check per entry,
  at most twenty inference calls, no retries or fallback to another provider.
- Preserve comparable output cap/deadline/settings where the route supports them;
  document differences, agent-wrapper effects and limitations before inference.
  A changed provider/protocol prevents a model-only causal conclusion.
- Report all ten outcomes, actual German card text, strict parsing and nested
  validation, semantic gaps, timing/usage and known reservation/billing limits.
  Do not weaken validators, silently repair answers, import or approve cards.
- If neither requested model or the authorized subscription route can perform
  this use case, report the concrete blocker rather than substitute a model.

## Verification Notes

- Audit original request/task identity and unchanged MiniMax baseline offline.
  Independently check captured outputs using the pinned application reader.
  Report unverified properties; no release or independent Latin authority claim.

## Implementation Notes

- 2026-11-28T13:31:17Z: Maintainer decision needed: supported Go route cannot run gpt-5.6-luna under the approved 20-call/60s bounded protocol; GPT-6 Luna is unreviewed in the launcher registry. Zero paid calls made.
- Discovery used only the launcher's authenticated catalog refresh and the
  public Go docs. GPT-5.6 Luna is reviewed and available; GPT-6 Luna is
  advertised but unreviewed. The bounded one-request, no-tool worker route
  allowlists only GLM-5.3-Flash, DeepSeek V4.1 Flash and Qwen3.8 Flash. The
  general agent route adds title generation and tool steps without a request
  cap or 60-second deadline, so twenty fresh tasks exceed twenty calls.
- Unblock only after the maintainer authorizes a scoped bounded-worker
  extension for reviewed GPT-5.6 Luna (one request, no tools or title call,
  8192 cap, 60-second deadline). No exception is approved; the skill and its
  allowlist are unchanged.
- 2026-11-28: The maintainer approved a scoped bounded-worker extension.
  The user skills repository now accepts explicit opencode-go/gpt-5.6-luna
  requests capped at 8192 output tokens and 60 seconds, with one broker
  request, no tools and title/summary/compaction disabled. Synthetic tests
  passed and an independent review found no blocking issue.
- Comparison ran 14:09-14:15 UTC on pinned executable c3a3523 with the
  unchanged MiniMax request texts and objectives: low effort, Responses
  protocol, one fresh generation and one answer-free blind check per entry,
  the check sent only after a strict-valid proposal. 16 of 20 calls, 16
  captured, no timeout, adapter or transport failure, no retries.
- Pinned strict reader: 10 of 16 valid; complete unresolved pairs for
  amica, puella, amare and laudare. Rejections: three outer JSON syntax,
  two nested source objects with the wrong keys, one nested source JSON
  syntax. No card was accepted, exported or imported.
- Contract findings shared with MiniMax: double-encoded nested evidence
  breaks JSON escaping; the literal stem-concatenation rule makes both
  models emit laudānt, which the reader accepts; the production-cue rule
  is resolved to the genitive and the template check accepts any cue.
- Reported usage: 41084 input and 23704 output tokens (4254 reasoning).
  Go quota after the run was 1% rolling, 0% weekly and monthly. Protocol,
  wrapper, deadline and retention differences prevent a model-only
  conclusion. An offline independent check reproduced all 16 strict
  outcomes; private outputs stay uncommitted.
- 2026-11-28T15:15:07Z: verification pass
