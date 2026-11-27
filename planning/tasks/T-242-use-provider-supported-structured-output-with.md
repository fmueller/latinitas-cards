---
id: T-242-use-provider-supported-structured-output-with
title: Use provider-supported structured output with explicit capability boundaries
status: completed
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-241-align-blind-model-tasks-with-whole-card-assertion
updated_at: "2026-11-27T04:19:43Z"
---

# T-242-use-provider-supported-structured-output-with Use provider-supported structured output with explicit capability boundaries

## Description

The current providers._request sends response_format=json_object to every OpenAI-compatible endpoint. This is JSON mode, not strict schema output. Adopt schema-constrained output where the exact provider/model supports it and accurately represent weaker or unverified guarantees elsewhere.

## Acceptance

- Verify current official OpenAI and MiniMax capabilities for the actual endpoint/model, not merely OpenAI SDK compatibility. OpenAI documents strict json_schema; the MiniMax M3.1-Flash Chat schema currently does not document response_format, while its separate native schema feature is documented for Text-01 only.
- Use a compatible strict schema on supported OpenAI endpoints/models, with explicit capability selection for compatible third parties. Do not send unsupported strict parameters to MiniMax or claim its accepted json_object parameter proves enforcement.
- Preserve strict local parsing and semantic validation, including refusal, truncation, malformed output and extra-key handling. No silent fallback, unbudgeted retry, invented support claim, or automatic policy enablement.
- Include the actual wire format in immutable request identity, accounting and budget preflight so schema growth cannot bypass reservations. Preserve existing ledger receipts and prior-run readability.
- Prove representative emitted requests and error behavior offline, document guarantees/limits and run the required review and final validation chain. Live qualification is separate and needs a valid bounded authorization.

## Verification Notes

- Official sources checked 2026-11-27: https://platform.openai.com/docs/guides/structured-outputs ; https://platform.minimax.io/docs/api-reference/text-chat-openai ; https://platform.minimax.io/docs/api-reference/text-post . Recheck if provider capabilities change.

## Implementation Notes

- Added explicit `openai-json-schema` capability selection without adding endpoint
  record keys or changing historical endpoint hashes. Supported configurations send
  strict closed/required schemas; local string/array and semantic checks remain.
- Known MiniMax Chat hosts omit undocumented `response_format` and reject strict
  selection. Existing non-MiniMax `openai` profiles retain JSON mode. Third-party
  compatibility and locally valid output do not prove remote schema enforcement.
- Extended complete-wire reservation and HTTP coverage; refusal/truncation/invalid
  output and a synthetic historical MiniMax transition retain fail-closed behavior,
  old charges, permission commitments and distinct current invocation identities.
- Dedicated simplifier and high-risk conventional review completed. Five specialist
  lanes plus independent security/data, claims/test-cost and overall verification
  closed the documentation overclaim and historical-transition evidence gap.
- Ten additional test cases, none removed. No live provider calls or qualification,
  semantic-policy enablement, Latin replay, study or collection changes occurred.
- RTK verified; exact evidence used proxy output. Comment-hygiene AST pass skipped
  because uncomment is absent. Mutation adapter doctor passed; its calibrated
  research-reader scope does not cover providers, so no provider score is claimed.
- 2026-11-27T04:19:43Z: verification pass
- 2026-11-27T04:19:43Z: Implemented explicit strict schema capability with preserved local validation and historical accounting; independent review and final Ruff/mypy/4013-pass pytest chain complete. Live qualification remains separate.
