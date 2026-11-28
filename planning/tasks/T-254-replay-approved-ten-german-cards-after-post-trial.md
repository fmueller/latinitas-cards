---
id: T-254-replay-approved-ten-german-cards-after-post-trial
title: Replay approved ten German cards after post-trial fixes
status: blocked
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-250-verify-integrated-post-trial-fixes-on-current
updated_at: "2026-11-28T12:21:27Z"
---

# T-254-replay-approved-ten-german-cards-after-post-trial Replay approved ten German cards after post-trial fixes

## Description

Execute the maintainer-approved R1 replay on accepted candidate 8b57008 using
the same ten German-instruction Latin entries and regenerate the existing trial
report. Approval: "Approve the replay under these limits and regenerate the
ten-card report". Pydantic AI stays deferred; use the current transport.

## Acceptance

- Preserve the ten selected identities and objectives from 01_Sprachen::Latein;
  exclude Stammformen, other decks and collection history. Fresh replay stores,
  no old answer reuse, Anki import, collection write or automatic approval.
- Both roles use MiniMax-M3.1-Flash, low reasoning, 8192 completion tokens,
  60-second deadlines and no retries. At most twenty calls, one generation and
  at most one answer-free blind check per entry, with fresh exact-wire grants.
- Reconcile the existing campaign and preserve all prior charges. Batch ceilings:
  800000 estimated tokens, USD 0.20 estimated reservation valuation and 1200
  request-deadline seconds. The disclosed prospective valuation is USD 0.189931
  for 774628 estimated tokens; these are not guaranteed provider-billing caps.
- Stop further dispatch on an unknown outcome pending reconciliation. No
  replacement cards, timing probes, fallback models or reruns until green.
- Report every selected entry, actual German card text, parsing/nested/direction
  and deterministic target/stem results, semantic gaps, timings, usage and retained
  reservations. Regenerate the existing Ten-entry Latin trial artifact; a negative
  or incomplete result is legitimate, not a reason for unapproved extra calls.
- Do not claim independent linguistic authority, protected T200 evaluation,
  native Anki acceptance or release readiness. T200/T201 remain untouched.

## Verification Notes

No executable product changes are planned. Validate exact request/scope hashes,
budget and artifact/ledger consistency offline, then independently check report
claims against saved outcomes. Use structural documentation checks for committed
task notes. Any product change requires separate scoped implementation and final
Ruff, mypy and pytest chain. Cite verification timestamps, not ignored paths.

## Implementation Notes

Replay paused on 28 November 2026 after six calls on approved candidate 8b57008.
Two calls succeeded (insula proposal and blind check), three proposals failed
malformed_output (domina, amica, via), and silva reached its deadline with an
unknown outcome. Entries six through ten were not dispatched. No abandonment,
retry, replacement, import, collection write or acceptance-policy change occurred.

Domina and amica violate the selected-sense contract at assertions.0. Via fails
outer answer JSON syntax at character 1033. Silva's worker-start marker is 382 ms;
elapsed time is 60009 ms, without observed headers or body. Remote cause and usage
remain unknown. Insula has complete assertion coverage but all four semantic
comparisons remain unresolved and authorize no acceptance.

The network-blocked offline audit passed on 28 November 2026: six request hashes,
five response hashes, strict-reader outcomes, blind task isolation, preserved ten
inputs/objectives/previews, original collection and all prior 52 campaign charges.
Retained batch reservations: 230446 estimated tokens, USD 0.056688 estimated
valuation and 360 request-deadline seconds. Captured usage: 13058 input and 13135
completion tokens, including 6097 reasoning tokens; unknown usage is not zero.
Campaign now has 58 development charges, including the unresolved silva charge.

Regenerate the existing Ten-entry Latin trial report as a paused checkpoint.
Resume requires provider reconciliation or explicit maintainer abandonment of the
unknown call with its reservation retained and no retry, before untouched entries
six through ten. Task remains blocked, not completed or release-accepted.
- 2026-11-28T12:21:27Z: Approved stop-on-unknown rule: silva generation timed out without captured headers/body; retain full reservation. Reconcile provider outcome or obtain explicit maintainer abandonment without refund/retry before dispatching untouched entries 6-10.
