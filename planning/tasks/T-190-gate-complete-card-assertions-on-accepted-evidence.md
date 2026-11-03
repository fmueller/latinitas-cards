---
id: T-190-gate-complete-card-assertions-on-accepted-evidence
title: Gate complete card assertions on accepted evidence
status: todo
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-188-check-mapped-deck-evidence-with-latin-rules-and
    - T-189-add-explicit-providers-and-blind-linguistic
updated_at: "2026-11-02T23:10:11Z"
---

# T-190-gate-complete-card-assertions-on-accepted-evidence Gate complete card assertions on accepted evidence

## Description

P50-2 shared full-assertion gate consumed by forms/phrases and orchestration. Own
deterministic policy evaluation, consumer integration and optional unresolved review
output. Follow approved contracts and spec execution/validation requirements.

## Acceptance

- A plausible supported entry can become exportable without human acceptance through
  approved library-only or optional LLM-supported paths. All asserted labels,
  translations, segmentations and explanations must be accepted.
- Matching answers with open objections, contradictions, unsupported explanations,
  malformed responses, stale bindings or failed checks withhold affected cards,
  leaving unrelated supported cards exportable.
- Preserve legitimate alternatives. No confidence/majority-vote shortcut or lowered
  standard for yield. Unresolved candidates stay outside default learning export;
  optional review cannot become mandatory homework or implicit approve-all.
- Automatic evidence cannot bypass the separate human gates for curated pairs and
  their actual rendered cards.

## Verification Notes

Check public preview/export with mixed accepted/withheld siblings and the negative
controls above. Include a no-human positive and a superficially agreeing unsupported
negative. Run the full mandatory validation chain.

## Implementation Notes
