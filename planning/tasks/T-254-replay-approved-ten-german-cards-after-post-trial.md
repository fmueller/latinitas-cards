---
id: T-254-replay-approved-ten-german-cards-after-post-trial
title: Replay approved ten German cards after post-trial fixes
status: completed
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-250-verify-integrated-post-trial-fixes-on-current
updated_at: "2026-11-28T12:51:37Z"
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

### Initial paused checkpoint

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
- 2026-11-28T12:43:55Z: Maintainer explicitly approved abandoning silva without retry/refund, retaining its full reservation, and continuing only untouched entries 6-10 under the existing limits.

### Completed continuation

After explicit maintainer approval, abandoned only silva through the supported
reconciliation API. Verified the charge count and its reservation unchanged;
remote processing and usage remain unknown. Continued only untouched entries six
through ten, once each, with the original exact-wire permissions and limits.
There were no further unknown outcomes, retries, replacements or extra probes.

Final replay outcomes: two complete pairs (insula, puella), seven malformed
proposals and the silva timeout. Twelve calls total: four strict-reader successes,
seven failures and one original unknown now abandoned in the campaign. Eleven
responses captured; seven pass the outer model, only four pass the strict reader.
All eight semantic relations from the complete pairs remain unresolved. No card
was accepted, exported or imported; this is replay completion, not release QA.

Domina, amica and bonus violate the selected-sense contract. Via has malformed
outer answer JSON. Amare, laudare and magnus encode nested source.entry as a
string instead of an object. Rejected laudare also renders laudant with a long a;
the report retains this output, not as an accepted spelling. Puella's checker
explanation conflicts with the generator about the ending added to puella's stem.
These content concerns remain distinct from syntactic/presentation validity.

Final network-blocked audit passed: twelve approved request hashes, eleven
response hashes and strict-reader outcomes, both blind comparisons, unchanged ten
inputs/objectives/regenerated previews and collection, and prior 52 charges.
Campaign now has 64 development charges and no unresolved outcome. Full retained
batch reservations: 464138 estimated tokens, USD 0.113862 estimated valuation and
720 request-deadline seconds, within the original limits. Captured usage: 29108
input and 28727 completion tokens, including 12628 reasoning tokens. Unknown-call
usage is excluded, not zero. The existing report is regenerated in place.

No executable product, dependency, test or runtime input changed. Reuse the final
Ruff/mypy/pytest evidence for executable c3a3523 (4054 passed, 7 skipped), followed
by the accepted integrated T250 receipt; only structural documentation checks and
the replay-specific offline audit are rerun here. Do not mislabel that reused gate
as a newly executed Python suite. Protected/native/migration/publication gates
remain unchanged. No further hosted work is authorized by unused capacity.
- 2026-11-28T12:50:46Z: verification pass

Independent read-only continuation review reproduced all eleven strict-reader
outcomes with networking blocked and found no material discrepancies. It checked
the nested failures, puella quotation, approved request hashes, unchanged initial
six calls, blind isolation, abandonment and accounting. Historical comparison and
collection hashes were outside its scope; those remain based on the earlier
report and coordinator audit. No new hosted inference was used for the review.
- 2026-11-28T12:51:37Z: verification pass
