---
id: T-180-include-known-lemma-deck-provenance-in-corpus
title: Include known-lemma deck provenance in corpus candidate reports
status: completed
priority: high
spec_ref: specs/v0.4.0.md#ranked-card-budgets
dependencies: []
updated_at: "2026-10-31T09:27:48Z"
---

# T-180-include-known-lemma-deck-provenance-in-corpus Include known-lemma deck provenance in corpus candidate reports

## Description

T-172's offline walkthrough on accepted runtime base
`cfdd4e2ed24819438e5c6bd0bea553cd4f0f2a00` found that the persisted
`corpus-cards select --known-lemmas PROFILE --known-lemmas-source SOURCE`
candidate report omits matching deck source identities. The binding
[ranked-card contract](../../specs/v0.4.0.md#ranked-card-budgets) requires
those informational links in both the report and preview, without changing
identity, eligibility or rendered card fields.

The synthetic Latin sentence `Marcus amicum videt.` has accepted morphology
for `amicum` with lemma `amicus`, a token gloss, a phrase translation and a
passage translation. The confirmed T-154 phrase deck/profile and its existing
source-scope sidecar supply `friend` / `reviewed-fixture-scope` for `amicus`.
Selection consumes the known lemma: lexical/guided novelty changes from 1 to
0 and scores from 220 to 200. Preview includes the matching source identity
and scope for both exercises. Neither candidate entries nor any other report
field contains those links; top-level `known_lemmas` holds only input digests.

`CandidateEntry` and report construction in `corpus_selection.py` retain no
deck provenance. Selection passes only `set(known.lemmas)` into ranking;
`corpus_output.py` separately resolves links for preview. Inspect these owners
and the strict candidates schema rather than adding a documentation workaround.

## Acceptance

- Persist matching deck source identities and their source scopes in the
  candidate report when known-lemma inputs match a candidate's accepted target
  lemma, following the existing homograph-suffix and fold contracts.
- Keep the candidates model, committed JSON Schema and user/agent contract
  documentation consistent. Do not amend the binding spec.
- Exercise real persisted CLI selection and preview with an asymmetric fixture
  containing matching and nonmatching targets. Assert independently expected
  source links and ranking values, not expectations derived from implementation.
- With and without known lemmas, preserve candidate identity, fact eligibility,
  rendered card fields and existing selection bytes. Changes to ranking values,
  report provenance and documented ranking outcomes remain permitted. Keep source,
  profile, manifest, annotations and claims bytes unchanged.
- Cover absent known-lemma inputs and a target with no matching deck identity;
  never invent provenance for sentence/phrase targets without a target lemma.
- Keep T-154 characterization goldens unchanged and run the exact mandatory
  ruff, mypy, pytest chain and Taskrail validation before verify/complete.

## Verification Notes

- Reproduced during T-172 on 2026-10-30. Main-agent persisted CLI evidence is
  retained in `/tmp/t172-evidence/journeys3/text/` and
  `/tmp/t172-evidence/known-preview.json` in the owning checkout's orb.
- At transfer, runtime changes and verification of this repair had not been
  performed.
- 2026-10-31: report entries now retain matching source identity/scope links;
  preview shares that provenance. Accepted featureless guided targets retain
  links while remaining withheld for `no_features`. Identity, ranking rules,
  selection state, rendered fields and deck inputs are unchanged.
- TDD reproduced the missing field and featureless-link omissions before their
  respective repairs. Dedicated simplification made no edits. High-risk
  conventional review used six specialist lanes, fresh boundary challenges,
  focused disposition verification and a fresh overall-outcome check. Its one
  substantive finding, missing featureless-target provenance, is fixed.
- Corrected frozen acceptance patch SHA-256:
  `5130aedd7392e19b7d985ee1b34f330fc5fdfd370c56e88c43b42b9b29bb6ee0`.
  Independent persisted-CLI QA passed 241 contract checks, 48 cross-revision
  checks and 26 subprocess exits. A separate public featureless replay and
  unresolved-target guard probes passed. Ordinary previews, selections and
  CSVs are baseline-identical; featureless guided output gains only the links.
- Final exact `uv run ruff check` → `uv run mypy` → `uv run pytest -v` chain
  passed: 3,202 tests passed, 7 skipped, one serializer warning. T-154 goldens
  remain byte-identical to their introducing commit. The binding spec and
  dependency lock are unchanged. Native Anki and integrated release-candidate
  acceptance remain with T-172/T-173, not this repair.

## Implementation Notes

- Created through `taskrail task new --spec-ref specs/v0.4.0.md#ranked-card-budgets
  --priority high`, not as a follow-up depending on T-172. T-172 must be rerun
  after the repair is independently accepted. T-173 remains a blocked
  maintainer-only gate.
- 2026-10-31T09:27:48Z: verification pass
- 2026-10-31T09:27:48Z: Reviewed and independently accepted provenance repair; T172 remains requeued and T173 remains a maintainer-only blocked gate
