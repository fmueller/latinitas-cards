---
id: T-228-verify-model-assisted-checks-in-acceptance-readers
title: Verify model-assisted checks in acceptance readers
status: completed
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-227-verify-source-grounded-semantic-comparison-proofs
updated_at: "2026-11-20T01:58:36Z"
---

# T-228-verify-model-assisted-checks-in-acceptance-readers Verify model-assisted checks in acceptance readers

## Description

Implement bounded model-assisted proof verification in AcceptanceReader under
specs/v0.5.0.md#shared-evidence-and-automatic-acceptance and adopted contract
sections 3–4. Reconstruct actual source-bound blind and semantic evidence using
the installed implementation. Diagnostic proofs remain diagnostic; proof
sufficiency and policy authority are separate gates.

Follow-up derived from T-227-verify-source-grounded-semantic-comparison-proofs's verification or discovery.

## Acceptance

- Verify complete actual blind-plus-semantic proof coverage, current source,
  binding, prompt, model, checker and comparison commitments, successful raw
  outputs, proposal-free actual requests and immutable stage relationships.
- Verify only the complete original card assertion set, retaining the original
  whole-task blind check and the cross-assertion no-trimming invariant. Proof
  structure verification is not card eligibility; target-only semantic success
  remains diagnostic and unsupported siblings continue to withhold the card.
- Reject blind-only authority, detached/recommitted artifacts, missing result
  discriminators, forged states, incomplete sibling coverage, contradictions,
  unsupported prose, origin mixing, stale evidence and proposal leakage.
- Exercise the real source-grounded verifier from saved synthetic offline calls
  through CLI/consumer boundaries; inspect proofs, decisions and empty receipts.
  Production policy stays disabled with empty scope and no authorization/history
  additions. Whole cards and ordinary learning export remain withheld.
- Preserve human/curated approvals and v0.4 German goldens. No reference
  acquisition, inference, judges, qualification/evaluation/feasibility, T200/T201
  lifecycle changes, release, publication or successor work.
- Use red/green TDD, dedicated simplification, independent risk-scaled review,
  verified dispositions and a read-only fixed-checkout integrated checkpoint.
  Run final Ruff → mypy → pytest before supported verify/complete; commit/push
  to main and confirm required exact-head CI success.
- Stop for an explicit decision before adding any consequential schema or
  authority extension absent from the adopted contract.

## Verification Notes

- Retain cumulative failures and dispositions in task-local working notes.
  Cite verification timestamps rather than ignored artifact paths here.
- A supported present_1s source-role proof is not lexical truth, general
  semantics, empirical qualification or whole-card acceptance. Broader producer
  prerequisites and endpoint full-input accounting remain outside this task.

## Implementation Notes

- Contract decision requested before executable edits: semantic_comparison.py
  emits one blind/transport check covering the full card, whereas its semantic
  checks are assertion-local. AcceptanceReader.decide_automatic prohibits
  trimming cross-assertion checks, and Acceptance rejects dangling assertion IDs.
  Sections 3–4 specify blind and semantic commitments but do not specify the
  partial-assertion adaptation rule.
- The overseer explicitly selected the contract-preserving complete-card route:
  retain the original whole-task blind check, add no assertion-local blind
  checks and leave target-only success diagnostic. The proposed assertion-local
  extension was not approved and will not be implemented. Safe verification,
  rather than artificial positive eligibility/export, is the task outcome.
- AcceptanceReader now regenerates the complete original blind/semantic Check
  set from current source-bound saved calls and the installed verifier. The
  pipeline retains those checks and their counter-stage root in immutable
  decisions. No callback, reference envelope or diagnostic authority flag is
  promoted to approval. Unsupported siblings and disabled policy still withhold
  whole cards; only the supported present_1s target is a diagnostic positive.
- High-risk independent conventional review covered general correctness,
  security/integrity, Python/tests and separate claims/comment accuracy. Nested
  delegation was unavailable, so the owner relayed coordinator assignments.
  One challenge round was consumed without resetting it for corrections.
- Verified corrections require actual indexed wrapper types, preserve human
  fallback before automatic index lookup, reject non-object proof indexes with
  structured stale-evidence decisions, and retain rule-only decision identity
  across processing-limit changes. Dedicated simplifiers made no edits; the
  coordinator source-verified all four dispositions. Existing resume criteria
  were not weakened. Final affected Ruff/mypy and seven tests passed.
- Cumulative failures are retained: initial missing API and origin-mixing reds,
  wrong-stage and human-fallback reds, malformed-index AttributeError red,
  fixture setup/type/receipt expectations, and the first broad run's two
  failures with 3821 passes and seven skips. That run was invalidated by later
  executable edits; stationary reruns distinguished a real rule-only revision
  regression from a changed-source failure that did not recur.
- Final read-only fixed-checkout checkpoint passed nine bounded offline
  journeys together in 56.46s on implementation digest
  6700c6620383723eeda71726d08ffd7bd2bcef2439d532c146c847f418842847.
  It regenerated complete actual request/output proofs and separately confirmed
  supported target semantics; rejected partial, omitted, stale, leaking,
  tampered, detached, deleted-result and malformed-index evidence; inspected
  stored proofs/decisions/empty receipts; and preserved production denial and
  genuine legacy human-reader authority. No injected passing Checks or enabled
  production policy supplied positive proof evidence.
- Checkpoint limits: the full CLI human fallback and rule-only processing-cap
  regression have separate owner/simplifier execution evidence. Synthetic
  transport is engineering-only, not endpoint accounting or qualification.
  Historical checkpoint fixture expectation failures remain retained.
- Final frozen-code gate, finished 2026-11-20T01:58Z: exact
  `uv run ruff check` → `uv run mypy` → `uv run pytest -v` passed;
  mypy checked 238 files and pytest reported 3823 passed, seven skipped,
  one existing serializer warning in 1335.69s. Source/test diff identity
  7490ac60346c903044162bbb428b4571d377f5603bfb130f9f689a1f0aa86494
  and implementation digest remained unchanged through the final gate.
  Formatting, diff whitespace and Taskrail structure checks also passed.
- This completes bounded verification capability, not card eligibility,
  reference truth, policy enablement or empirical qualification. Delivery and
  exact-head CI evidence are recorded in the thread report; checkout retained
  for corrections. No successor is selected.
- 2026-11-20T01:58:36Z: verification pass
- 2026-11-20T01:58:36Z: Bounded reader capability implemented and independently checkpointed; complete-card no-trimming and disabled-policy/export controls retained. Target-only success remains diagnostic. Final local gate passed; delivery CI tracked in thread report. No successor selected.
