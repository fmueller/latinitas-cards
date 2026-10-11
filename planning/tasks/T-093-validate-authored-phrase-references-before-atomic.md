---
id: T-093-validate-authored-phrase-references-before-atomic
title: Validate authored phrase references before atomic review writes
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#reviewed-claim-decisions-for-all-generated-cards
dependencies: []
updated_at: "2026-10-11T20:06:20Z"
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

- Authoritative guard: fetched origin; local HEAD and origin/main both matched
  `f76a9fa7076d22d67192704017cdca0a64424f95`. Pinned v0.2.1 spec hash matched
  `0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4`.
  Independent `taskrail validate` returned `state valid`; `next --json` selected
  T093 before `start`. No product transfer or conflicting task was present.
- Owner contract: phrase answer bindings supply the existing typed `Reference`
  model; `record_batch` validates incoming authored JSON before constructing
  proposed entries or atomic replacement. Existing downstream validation and
  ambiguity withholding are unchanged; no import-cycle wrapper was added.
- Strict public CLI RED: `uv run pytest tests/unit/phrase_inputs_test.py -k
  invalid_authored_reference -q` failed both cases on the unchanged recorder:
  invalid mixed batch exited 0 with `recorded: 2`; individual exited 0 with
  `recorded: 1`. Initial test setup used an obsolete corrected fingerprint;
  refreshing through `claims list` exposed the actual reported failure.
- GREEN: the same regression covers valid translation correction with optional
  inapplicable features omitted, empty-features rejection, indexed diagnostics,
  byte-identical prior sidecar, retained eligibility and identical CSV reruns.
  Focused phrase-input/recipe/review-command tests: 95 passed. Existing stale,
  conflict and contextual ambiguity coverage remained green.
- Dedicated Task simplifier loaded `code-simplifier`: no edits recommended;
  focused regression 2 passed. No suggestions were rejected.
- Separate parallel read-only Task lanes loaded `code-reviewer`: General loaded
  `code-reviewer.md` (no companion); Security loaded `security-reviewer.md`,
  security-review reference and cloud-infrastructure resource; Python loaded
  `python-reviewer.md` and python-patterns reference. These cover correctness,
  atomic persistence/data integrity, input trust boundary and Python typing.
  Database omitted: no SQL/schema/migration changes; framework/domain lanes
  omitted: no web framework or new linguistic model; two specialists fit budget.
- Verbatim lane candidates: General "No concrete task-relevant findings."
  Security "No concrete task-relevant findings." Python "No concrete
  task-relevant findings." Fresh candidate-validation Task loaded
  `code-reviewer`, accepted all three conclusions; finding IDs, rejected IDs,
  deduplications and dispositions were empty. No fixes or deferrals required.
- Fresh disposition-verification Task loaded `code-reviewer` and General,
  Security/Python references: "No concrete task-relevant findings." One review
  cycle; no unresolved or newly introduced task-relevant issues.
- Real public executable replay: `uv run latinitas-cards claims record` with the
  mixed batch exited 1, `recorded: 0`, invalid item 1 identified
  `readings.0.features`. Sidecar SHA-256 before/after:
  `2e8c8d689fbd49062f527dea6b04cabf419fdacbdb416f600fb2a6b8db8d50d6`.
  `claims inspect --check` exited 0, eligible true, no diagnostics;
  `phrases export` exported 1. Same canonical-source-path CSV before/after:
  `9a33c5a5f692c3ff7dda3ef09852a4ebde44f045ce48f1c6d79fb59762d43953`.
  A preliminary comparison used a symlink source alias against the canonical
  source baseline, changing source identity; the same-path replay passed `cmp`.
- Asymmetric real CLI controls: individual valid correction to "gemeinsam mit
  einem Freund" recorded 1, check exited 0/eligible true and export included
  that translation. Valid alternate dative/ablative readings recorded 1 but
  check exited 1/eligible false with "The tested distinction is unresolved in
  the supplied contextual readings." Structural validity is not teachability.
- Final exact chain: `uv run ruff check` (All checks passed!), `uv run mypy`
  (no issues in 107 source files), `uv run pytest -v` (1045 passed in 42.81s).
  Fresh disposition verification followed these checks before task finalization.
- Scope: T093 only; T094 remains separate. No follow-up discovered. Native
  Desktop/AnkiMobile acceptance remains maintainer-deferred (2026-10-11), pending
  and not passed or covered. ROUND2/5 is not clean; this fix alone does not
  authorize ROUND3 or alter that gate. Existing CSV/manual/native limits stand.
- 2026-10-11T20:06:20Z: verification pass
- 2026-10-11T20:06:20Z: Implemented and independently reviewed T093 only; final ruff/mypy/pytest chain passed. Native acceptance pending; ROUND2/5 not clean; T094 separate.
