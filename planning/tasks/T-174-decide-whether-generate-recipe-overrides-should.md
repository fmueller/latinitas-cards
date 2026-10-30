---
id: T-174-decide-whether-generate-recipe-overrides-should
title: Decide whether generate --recipe overrides should stale reviewed claims
status: completed
priority: medium
spec_ref: specs/v0.4.0.md#german-and-english-output
dependencies:
    - T-157-localize-fixed-card-strings-and-english-reviewed
updated_at: "2026-10-30T19:36:55Z"
---

# T-174-decide-whether-generate-recipe-overrides-should Decide whether generate --recipe overrides should stale reviewed claims

## Description

Follow-up found during T-157 review; the behavior already exists at db90405 and is not
caused by T-157. `generate --profile P --recipe principal_part_completion` (one recipe
instead of the profile's two) reports every stored decision as "Stale or unmatched
decision ignored" (16 for the v0-4-0 de-light-user fixture), while `preview` without
`--recipe` applies them. `commands/principal_parts.py` writes `selected_recipes` into the
profile override, and claim fingerprints bind the whole deck profile except `morphology`
(`claim_review.py`). Section `#german-and-english-output` restates that fingerprint
contract as unchanged since v0.2.1, so changing the fingerprint inputs needs a maintainer
decision: either a run-time `--recipe` filter is excluded from claim binding, or the
current staleness is documented as intended.

## Acceptance

- The maintainer records which behavior is intended for a `--recipe` override.
- If the binding changes, existing German claim fingerprints and claims-v1 / v0-4-0
  goldens stay byte-identical for runs without `--recipe`, with a regression test for a
  single-recipe run that keeps accepted decisions.
- Otherwise the `--recipe` help and docs state that narrowing recipes stales reviews.

## Verification Notes

- RED: both single-recipe CLI regression cases failed on the unchanged implementation
  because all 16 decisions became stale. GREEN: both retain 4 accepted label decisions,
  12 withheld optional claims, and four eligible selected-recipe cards for amō.
- The unfiltered CLI export SHA-256 captured on unchanged remote main is
  `04eb85e86efc485638ee0912db8dc08955ed54f365aa54de4c055ff542d1badd`;
  the regression checks those exact bytes and saved-profile recipe-edit staleness.
- Strengthened the two cases to compare all 16 exported decision fingerprints/statuses
  with the unchanged sidecar and check successful filtered previews and immutable
  checkpoints. A disposable negative control corrupting only withheld fingerprints
  failed both cases at the new mapping assertion; the real implementation passes.
- Dedicated simplifier: no changes in either pass. High-risk conventional review used
  separate general, Python/data-integrity, test/golden and prose reviewers, followed
  by fresh integrity and overall-outcome adversarial verifiers. All required lanes
  approved; no confirmed runtime findings. The withheld-binding and positive-preview
  coverage suggestions were fixed and independently verified. Manifest/package filter
  forwarding was source-traced; an exhaustive new input-mode matrix was not claimed.
- Independent integrated QA derived expectations from the decision/spec before reading
  implementation reports. On clean disposable data, 81 real CLI calls passed 271
  assertions (including 46 golden checks), 12 exact stdout comparisons and 12 exact
  sixteen-decision mapping comparisons. Controls covered accepted/withheld labels and
  translations, both narrowed recipes and the original set, saved recipe/language
  edits, another binding-changing CLI override, and archived-base unfiltered bytes.
  The final QA rerun passed the same matrix; native Anki acceptance remains untested.
- Reviewed and independently QA-tested cumulative candidate: base
  `5a73759f8320d794749a475f46460b304bbdb759`, binary-diff SHA-256
  `c00b4fc24da8cf1f0e1bc00570dd93d6205955cafaec6c1d99cc04a58c835c3c`.
  Only these verification notes and Taskrail lifecycle metadata change afterward.
- Final exact chain: `uv run ruff check` passed; `uv run mypy` passed (198 source
  files); `uv run pytest -v` passed (3171 passed, 7 skipped, one existing negative-test
  serializer warning, 414.05 seconds). `mise run setup` installed the pinned hooks.

## Implementation Notes

- 2026-10-29T05:30:24Z: maintainer decision: whether a generate --recipe override should be excluded from claim fingerprints, which changes the unchanged v0.2.1 fingerprint contract
- 2026-10-30T19:12:09Z: Maintainer approved: CLI --recipe narrowing is an output filter that preserves existing reviews using the original saved-profile fingerprint; saved-profile edits retain review-binding rules and unfiltered German goldens stay byte-identical.
- Scope clarification: preservation applies to runs without `--recipe`. The maintainer
  explicitly authorized only filtered CSV/checkpoint expectation changes caused by
  honoring accepted/withheld reviews. No fixture input, sidecar, unfiltered output,
  claim fingerprint algorithm or unrelated Fixed Contract may change.
- Recipe selection travels separately to rendering; claim construction and comparison
  continue using the original profile. Other explicit profile overrides retain their
  existing binding rules. Rendering provenance still describes the selected recipes.
- Golden history: T-154 captured unchanged output with `regen-legacy-goldens.py` and
  records SHA-256 values in CAPTURE. The full capture entry point intentionally refuses
  modified source; no broad recapture was run. A disposable allowlisted replay uses
  the existing `claim_case` harness to update only the completion CSV and the two
  filtered checkpoints, then their three CAPTURE hashes. Expected card keys were
  independently specified: four accepted amō roles, including supine. The later
  recognition export stays empty because prior completion slots cannot be retired
  by CSV; its checkpoint retains those four completion slots. All other bytes must
  compare unchanged. Independent review verified every changed golden as necessary.
- 2026-10-30T19:36:55Z: verification pass
- 2026-10-30T19:36:55Z: Implemented approved output-only recipe filtering; final checks and independent review/QA passed. Verification run on 2026-10-30; no successor claimed.
