---
id: T-093-validate-authored-phrase-references-before-atomic
title: Validate authored phrase references before atomic review writes
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#reviewed-claim-decisions-for-all-generated-cards
dependencies: []
updated_at: "2026-10-11T19:29:13Z"
---

# T-093-validate-authored-phrase-references-before-atomic Validate authored phrase references before atomic review writes

## Description

ROUND2 medium-severity violation of documented atomic invalid-item rejection,
confirmed on 45a7cbfb7a73db3aab77ab838f94423829dfdfe6. A correction whose complete
Reference JSON has `readings[0].features = {}` returns success from claims record
and replaces a previously usable accepted answer. Preview subsequently reports
Reference validation errors and withholds the exercise. Downstream withholding
works; this is not an invalid-answer export finding.

From a fully reviewed documented phrase workspace, list current exercise claims.
Copy an accepted combination's exact reviewer/reason for an idempotent batch item.
For an answer item, copy its Reference value and set its first reading's features
to an empty object, retaining all other fields and adding explicit authored
evidence. Run `claims record inputs.json --selections selected.json --recipes
recipes.json --batch batch.json --json`. Actual: exit 0, records 2, sidecar changes.
Expected: nonzero item-level diagnostic and byte-identical sidecar. Subsequent
`claims inspect ... --check --json` returns 1; export now omits the affected note.

Dedup: T064's completed atomic-invalid-batch contract is explicitly reused by
T087 phrase answers, but this typed Reference validation gap is not separately
tracked. T090/T091 address different surfaces and pass their original regressions.

## Acceptance

- Validate incoming authored phrase References against their supported structural
  schema before replacing the review store, for individual and batch recording.
- A valid item mixed with an invalid Reference rejects atomically with an actionable
  item-level diagnostic and byte-identical prior sidecar; previous eligible answers
  remain usable. Cover an empty features object and a valid corrected Reference.
- Preserve distinct legitimate contextual ambiguity/withholding semantics: do not
  conflate structurally invalid input with a valid but unteachable distinction.
- Existing stale/conflicting batch rejection and downstream withholding still work.

## Verification Notes

- Two independent clean-workspace replays on 2026-10-11: initial inspect/record/
  subsequent inspect/export exits 0/0/1/0. Sidecar SHA-256 changes from
  `2902b72446e20903d2d9c117a1fc348782d5007bac252589bcb18b3f5e9ffd61` to
  `8a55ef16955f9815e18fa16f9ad6b4aed836d0f1bd1086221b1cdfc4c610c909`.
- ROUND2 synthetic bundle contains exact batch, before/after sidecars and commands
  in `supervisor-replay/empty-features`; `confirm_findings.py` repeats the boundary.
- Source spec SHA-256:
  `0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4`.
- Tester filed only; no implementation or task verification performed.

## Implementation Notes
