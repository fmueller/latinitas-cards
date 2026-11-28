---
id: T-257-simplify-small-model-prompts
title: Simplify single-card model prompts for smaller models
status: completed
priority: high
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-254-replay-approved-ten-german-cards-after-post-trial
updated_at: "2026-11-28T15:37:27Z"
---

# T-257-simplify-small-model-prompts Simplify single-card model prompts for smaller models

## Description

Analyze and implement the smallest prompt simplification that addresses the
observed small-model contract failures, using implementing-changes. Keep transport
retry changes owned by T256 and the baseline Go comparison unchanged in T255.

## Acceptance

- Inspect the actual one-card request builders and latest captured failures.
  Domina/amica/bonus put selected_sense on non-translation assertions; amare,
  laudare/magnus use string source.entry where an object is required; via has
  malformed JSON. Determine which instructions are redundant or contradictory.
- Keep fresh single-card prompts and answer-free blind checks. Simplify the
  model-facing contract without deleting required source/sense/reference evidence,
  weakening nested validators, acceptance semantics, privacy or accounting.
- Make required field shapes clear, including selected_sense and nested values.
  Prefer removing redundant material over adding a second competing contract.
  Version changed prompts/requests so old permissions/reuse cannot silently apply.
- Use captured outputs and synthetic/offline tests during implementation; no live
  MiniMax or other provider inference is authorized. Do not claim hosted small-model
  reliability has improved without a separately measured comparison.
- Complete independent review, final repository checks and Taskrail lifecycle;
  commit and push on main when ready. No migration, automatic approval or release.

## Verification Notes

- Show before/after representative request sizes and preserved task/evidence
  contracts, with focused regression checks. Follow the full implementing-changes
  workflow and final ruff/mypy/pytest chain. Subscribe to CI without waiting on it.

## Implementation Notes

Offline analysis rebuilt the twelve captured T254 MiniMax requests with
prepare/ingest only; all were byte-identical to the captured bodies. Causes found
in the actual request: the form objective's top-level `selected_sense: "source"`
reused the answer field name and the `source` assertion id (domina, amica and
bonus put it on the source label); the source value contract said only "the
supplied entry" (amare, laudare and magnus returned the lemma string); via broke
its outer JSON while re-quoting whole nested evidence JSON; the selected-sense
rule was stated twice for MiniMax.

Changes (prompt revision proposal/3 and blind/3): one shared answer contract
states the selected-sense rule once for every provider and points to each
value_contract; the form objective names the selected sense `sense_key`; the
source contract names the whole entry object and its selected sense object;
citations quote only the relevant part of supplied evidence; feature terms are
sent only to recognition tasks. Validators, permissions, accounting, dispatch,
retry, disclosure, source evidence and the answer-free blind task are unchanged.

Rebuilt request bodies: 10990-11686 to 9854-10550 bytes per proposal; 223474 to
200754 bytes over all twenty role requests. The strict reader still gives the
captured responses their original outcomes. No hosted inference was used; any
hosted reliability effect is unmeasured.

T255's shared Luna/MiniMax gaps against this change: the nested `source` shape and
re-escaped nested evidence are addressed only by clearer prompt text (unmeasured).
Still follow-up product/validator work, outside this task: the double-encoded
string `value` format; `laudā + nt` forcing `laudānt` under literal-concatenation
segmentation, which the strict reader accepts without comparing the preview; and
production cues resolved to the genitive (`dominae`, `puellae`), which the
template check accepts as any nonempty cue.

Final gate on T256 8e7d463 plus this diff: ruff and mypy passed; pytest 4059
passed, 7 skipped in 3568 seconds. Rebased afterwards onto planning-only T255
7753c00 with the executable diff unchanged (sha256 ffe77ed5).
- 2026-11-28T15:37:27Z: verification pass
