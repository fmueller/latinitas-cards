---
id: T-242-use-provider-supported-structured-output-with
title: Use provider-supported structured output with explicit capability boundaries
status: todo
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-241-align-blind-model-tasks-with-whole-card-assertion
updated_at: "2026-11-27T01:28:37Z"
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
