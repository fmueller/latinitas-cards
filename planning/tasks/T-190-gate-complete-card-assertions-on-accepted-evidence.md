---
id: T-190-gate-complete-card-assertions-on-accepted-evidence
title: Gate complete card assertions on accepted evidence
status: completed
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-188-check-mapped-deck-evidence-with-latin-rules-and
    - T-189-add-explicit-providers-and-blind-linguistic
updated_at: "2026-11-18T07:08:44Z"
---

# T-190-gate-complete-card-assertions-on-accepted-evidence Gate complete card assertions on accepted evidence

## Description

P50-2 shared full-assertion gate consumed by forms/phrases and orchestration. Own
deterministic policy evaluation, consumer integration and optional unresolved review
output. Follow approved contracts and spec execution/validation requirements.

## Acceptance

- Implement the no-human route through approved library-only or optional LLM-supported
  policies, tested positively with an isolated synthetic application registry. All
  asserted labels, translations, segmentations and explanations must be accepted.
  With the production enabled set empty, even all-checks-pass automatic candidates
  stay withheld and outside ordinary learning export/destination handoffs. Stored
  acceptance, stage/export commands, resume and old batches cannot bypass authority.
  Actual automatic enablement needs T-200 evidence and separate maintainer approval.
- Matching answers with open objections, contradictions, unsupported explanations,
  malformed responses, stale bindings or failed checks withhold affected cards,
  leaving unrelated supported cards exportable.
- Preserve legitimate alternatives. No confidence/majority-vote shortcut or lowered
  standard for yield. Unresolved candidates stay outside default learning export;
  optional review cannot become mandatory homework or implicit approve-all.
- Automatic evidence cannot bypass human approval of curated rules/explanations and
  actual rendered cards, or language-specific mnemonic approval whenever one is shown.
  Rule-only cards do not require absent mnemonic content or approval.

## Verification Notes

Check public preview/export with mixed accepted/withheld siblings and the negative
controls above. Include an isolated-fixture no-human positive, an all-checks-pass but
unenabled negative and a superficially agreeing unsupported negative. Synthetic
policies must not ship enabled or count as linguistic evidence. Run the full mandatory
validation chain.

## Implementation Notes

- Added the opt-in `card_acceptance` library boundary: current full assertion
  matching, complete decision coverage, human/automatic/mixed provenance,
  independent sibling selection and optional unresolved inspection. Inspection
  grants no approval and unresolved cards stay outside learning selection.
- `AcceptanceReader.decide_automatic` consumes the complete source-analysis check
  set to create fresh card-specific automatic decisions. Every intersecting check
  remains relevant; checks spanning omitted assertions cannot be sliced. Original
  check envelopes and recursive prerequisite commitments remain mandatory.
  Stored aggregate decisions and human attestations remain atomic.
- Independent high-risk review reproduced shared-analysis over-withholding. The
  actual deck-analysis regression now exports supported recognition/label siblings
  under an isolated synthetic policy while withholding an unsupported explanation.
  Production-disabled authority withholds all automatic cards. Unrelated report
  corruption does not veto supported siblings; required source corruption does.
- Synthetic fixtures cover every assertion class, matching-but-unsupported answers,
  relevant objections, missing/ambiguous artifacts, stale bindings, old-batch
  withdrawal and human-sibling survival. They are engineering tests, not linguistic
  evidence. Production policy records remain unchanged with no enabled scope.
- Consumers must supply current bindings/assertions/versions and enumerate every
  rendered linguistic claim, with the complete analysis check set. T-191–T-193 own
  concrete recipe/rendering/orchestration writes and repeat validation at output
  and destination handoff. Retained readers do not rediscover changed source data.
- Generic curated attestations are denied rather than substituted for rule,
  rendered-card or language-specific mnemonic fingerprint approvals. T-196 owns
  that dedicated adapter; rule-only content does not gain a mnemonic requirement.
  Model-method acceptance is not implemented by this library-only route. T-200
  empirical/enablement and T-201 real-model/native gates remain unsatisfied.
- 2026-11-18T07:08:44Z: verification pass
- 2026-11-18T07:08:44Z: Task-local library-only implementation verified after independent high-risk review and final Ruff/mypy/pytest gate (3514 passed, 7 skipped). Production policy remains disabled; no empirical, integrated or release acceptance claimed. No successor selected.
