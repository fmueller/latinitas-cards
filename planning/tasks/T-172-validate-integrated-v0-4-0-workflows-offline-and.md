---
id: T-172-validate-integrated-v0-4-0-workflows-offline-and
title: Validate integrated v0.4.0 workflows offline and prepare the 0.4.0 candidate
status: completed
priority: low
spec_ref: specs/v0.4.0.md#goals
dependencies:
    - T-170-document-corpus-first-tei-and-english-workflows
    - T-171-prepare-the-v0-4-0-native-acceptance-kit
    - T-159-preserve-tei-notes-apparatus-and-editorial-markup
    - T-161-import-aligned-parallel-translations-as-annotation
    - T-169-plan-managed-updates-for-corpus-exercises
    - T-167-deselect-and-restore-selected-corpus-exercises
updated_at: "2026-10-31T10:29:56Z"
---

# T-172-validate-integrated-v0-4-0-workflows-offline-and Validate integrated v0.4.0 workflows offline and prepare the 0.4.0 candidate

## Description

Delivery Order step 9. An offline end-to-end walkthrough using only documented CLI
commands in a temp dir outside the checkout, then the 0.4.0 candidate (version, lock,
changelog under `[Unreleased]` moved to 0.4.0, readiness doc, README). No tagging, release
or publication; no runtime code changes. Native evidence belongs to the gate.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- At start and before verify, `taskrail status --json` shows active spec
  `specs/v0.4.0.md` with only this task and the blocked gate open; otherwise block or
  requeue as T-124 did for v0.3.0.
- Walkthrough: TEI fixture and plain-text corpus normalize and validate; segmentation;
  `latinitas-morph/1` and translation patches (one via `import-translations`) through
  plan/approve/apply; `claims record` acceptances; `corpus-cards profile validate`,
  `select` with known lemmas, `preview`, `deselect`/`--restore`, `export`,
  `inspect --check`, `plan`; an English deck profile preview.
- Negative controls: applied but unreviewed translations give no eligible card; a passage
  correction gives `context_changed`; a tampered CSV fails `inspect --check`; an unknown
  TEI element exits 3; a `fr` corpus profile exits 3.
- The walkthrough also runs `corpus-cards select` and `preview` on the TEI corpus; shows
  that adding the known-lemma input changes only ranking values and report deck links
  (card fields and eligibility identical); and a test asserts no module under `src/`
  compares translation text for grading (no string equality on translation values
  outside rendering and validation).
- Characterization goldens from T-154 are unchanged since the commit that added them.
- TDD: update the release-metadata test first (RED on the old version), then set
  `project.version` and the `uv.lock` entry to 0.4.0, move `[Unreleased]` entries under
  `## [0.4.0] - YYYY-MM-DD` with comparison links, and add `docs/release-v0.4.0.md`
  (scope, walkthrough, limits, native evidence pending the gate). T-173 may correct only
  the date before tagging.
- Run `UV_OFFLINE=1 bash scripts/check-dist.sh` and `bash scripts/check-install.sh
  "$(uv python find 3.14)"` as T-124 did (rerun without `UV_OFFLINE` if only the cache
  is missing; record "not run: offline" if no network).
- No tag, release, upload, workflow dispatch, spec activation or runtime change. A
  product defect found here is filed with `taskrail task new --spec-ref
  specs/v0.4.0.md#<anchor> --priority high --title "..."` (never `--follow-up` on this
  task), then this task is requeued with `taskrail block <id> --reason "requeued behind
  T-NNN"` and `taskrail unblock <id> --reason "requeued behind T-NNN"`; uncommitted
  candidate edits are discarded first.
- All goldens and the full mandatory chain pass; `mise run check` and `taskrail validate`
  pass.

## Verification Notes

- 2026-10-30 attempt on accepted `origin/main`
  `cfdd4e2ed24819438e5c6bd0bea553cd4f0f2a00`: setup passed; initial
  Taskrail validation/status/selector confirmed this task plus blocked T-173
  were the only open active-spec work. No parent-local state was transferred.
- The documented TEI walkthrough ran outside the checkout and produced four
  eligible recipes, individual claim acceptances, deselect/restore, initial
  export and successful `inspect --check`. Plain-text normalization, validation,
  segmentation, imported translation patch plus morphology/gloss/phrase facts,
  plan/approve/apply and individual claim decisions also passed.
- Blocker: adding the confirmed known-lemma fixture changes lexical/guided
  novelty from 1 to 0 and scores from 220 to 200, but the candidate report has
  no matching deck-source links. Preview with the same inputs reports
  `friend` / `reviewed-fixture-scope` for both targets. Filed high-priority
  T-180 against the real ranked-card anchor; no runtime repair attempted.
- Release metadata test first failed as expected (`0.3.0` versus `0.4.0`).
  All candidate metadata and test edits, including the draft semantic AST
  no-grading guard, were then discarded before requeueing. The draft diff and
  CLI transcripts remain in `/tmp/t172-evidence/` in this checkout's orb.
- Full final ruff/mypy/pytest chain, `mise run check`, packaging/install checks,
  candidate simplification/review and final candidate checkpoint were not
  reached. No verify/complete, commit, push, tag or release occurred. The
  independent blocker checkpoint is separate from candidate acceptance.
- T-154 goldens match their introducing commit `681cf22e672ce595efeac418d859509569bc9807`
  byte-for-byte; T-174's authorized filtered claims-v1 updates are a distinct
  fixture set. No goldens were recaptured. Native Linux Anki Desktop acceptance
  remains pending T-173; AnkiMobile remains unverified and outside that gate.

## Resumed candidate execution — 2026-10-31

- Accepted cumulative base:
  `3a20e48c7ad1bdee8550422482f85610b8ee8f80`, containing completed T-180
  and dependency advancement `773664893a1f8c6705bc0ff9c6469a7e34f0119d`.
  Compared delivered T-172/T-180 notes with retained local planning edits,
  archived the obsolete duplicates, then fast-forwarded. No parent STATE was
  transferred. Setup and the active-spec-only start guard passed: v0.4.0,
  zero active, only T-172 todo and T-173 blocked open; off-spec T-128/T-179
  were unchanged. Only T-172 was started.
- Main walkthrough passed 89 persisted public CLI calls outside the checkout,
  recorded under `/tmp/t172-evidence/resumed-journeys4/commands.json` with
  the reproducer `/tmp/t172-evidence/walkthrough.py`. Both text and TEI
  lanes validate and segment, import a parallel translation patch, add reviewed
  morphology/gloss/phrase facts through plan/approve/apply, record individual
  claims and select/preview all four recipes. Before claim acceptance all four
  candidates were ineligible with `open_fact`; applying a patch alone was not
  accepted as review.
- Known-lemma inputs now produce matching `friend` /
  `reviewed-fixture-scope` provenance in reports and previews. Rankings change,
  while keyed exercise fields, eligibility and selection bytes stay unchanged;
  copied known deck/profile/manifest bytes stay unchanged. Deselect/restore,
  export and CSV inspection passed in both lanes. Tampered CSV inspection exits
  1 with `card_drift`, French corpus profiles exit 3, and an unknown TEI element
  exits 3. Corrected source facts were explicitly re-reviewed after archiving
  obsolete claim decisions; the three unchanged exercise keys report
  `context_changed`, not silently transferred context. The phrase target changes
  identity when its folded wording changes, as specified.
- Corpus planning uses explicitly synthetic offline snapshots, not native
  evidence: public `managed adopt`, `corpus-cards plan` and `managed plan`
  produce four retained effects with zero update operations on unchanged cards.
  The English APKG deck profile preview shows English fixed labels. This proves
  neither native import nor scheduling/personal-data preservation.
- Metadata TDD was repeated on this base: updated release test failed on the old
  `0.3.0`, then passed with project/lock `0.4.0`, dated changelog, empty
  Unreleased, candidate readiness record and README pointer. The 91 non-project
  lock records are unchanged. The walkthrough documentation now reflects T-175's
  existing exclusion of exact token glosses from phrase candidates; no runtime
  change was needed.
- `UV_OFFLINE=1 bash scripts/check-dist.sh` passed. Wheel SHA-256:
  `063e00e1408c394b5b1e76246183ca584a834c6d5868e199dc4037a42b3d9daf`;
  sdist SHA-256:
  `2d13068cab0d9cbd373d496ebaf57529b2a5803637648ebb80dbd3227254d77b`.
  `bash scripts/check-install.sh "$(uv python find 3.14)"` passed both artifacts
  on Python 3.14.7 outside the checkout: help/version, USFX normalize/validate,
  and absence of optional CLTK/Stanza/Torch distributions. Distribution building
  was offline; installation used the command's normal resolver/network policy,
  not a claim of fully offline dependency installation.
- Initial mandatory chain stopped at a missing type annotation in the new test
  helper; corrected it and restarted the entire chain from Ruff. Final chain,
  simplifier/review dispositions and frozen independent checkpoint are recorded
  below when completed. No candidate acceptance is claimed by this intermediate
  record; native evidence remains pending T-173.

## Review and final independent checkpoint

- Dedicated `code-simplifier` removed only the temporary-file round trip from
  AST detector controls; all controls and source inventory remained intact.
  Main rechecked Ruff, mypy and the 31 targeted corpus/release tests. The
  source-wide guard examines all 97 Python modules; its 17 exact reviewed
  equality occurrences are validation/conflict/drift/metadata checks, not
  learner grading. Its nine injected positive/negative controls include a
  renamed alias and string normalization. Arbitrary interprocedural dataflow
  is explicitly outside this bounded guard.
- Independent conventional review used a medium-risk coordinator. Nested
  delegation was unavailable, so main relayed separate read-only General
  (`code-review`), combined Python/test and substantive prose (`comment-audit`)
  specialists back to the same coordinator. General and Python found no
  defects. The sole prose finding was that "Source corrections retain identity"
  failed to distinguish passage identity from target-dependent exercise IDs.
  A fresh adversarial verifier confirmed that minor non-blocking finding and
  challenged the overall verdict; no new findings emerged. Fixed the wording
  and had the coordinator independently verify its disposition against spec
  and source. No required findings remain. One challenge round, no reset or
  follow-up challenge; wording-only readback, not a new specialist fan-out.
- Optional tests for deliberately unsupported analyzer flows and another
  changed-target identity case were not required fixes and were not added.
  Existing identity vectors, main persisted phrase correction and independent
  targeted identity checks cover the relevant contract; no runtime behavior
  changed. There are no deferred required fixes or unimplemented successors.
- Frozen cumulative review patch SHA-256:
  `74666a5a734216c74e210e31ca806214e29f624fc7ef46adc649515995a32cdf`.
  After the sole prose correction:
  `926a84a2109b8b82d8920c12d2296910f45bdf8a35f9f9f7bf5cade39ece6903`;
  corrected manifest SHA-256:
  `ce5262ac468e9b22392995aedce6a37e1f9568ecf84b6250872264073e2bc753`.
  All 11 actual files matched the corrected manifest. Later additions are
  planning/evidence/lifecycle records only, separately reconciled before close.
- A separate final QA subagent independently derived expectations before
  execution and passed 157 persisted public-CLI calls: 106 source-executable,
  33 fresh installed-wheel and 18 planning/correction calls. It used its own
  `Puella rosam portat.` fixtures, hand-computed asymmetric weights/scores,
  independently hashed identities and source-state/CSV byte comparisons. It
  covered all required text/TEI journeys and negatives, known report/preview
  provenance/invariance, accepted featureless guided morphology (`no_features`
  with retained links; lexical eligible), and unresolved morphology with no
  invented links. The source correction retained four keys as `context_changed`;
  explicit confirmation restored eligibility. Zero live claim decisions were
  deleted during the documented selective obsolete-decision migration.
- Final QA independently installed the wheel outside the checkout, confirmed
  its import path in site-packages and matched all 97 wheel Python files to
  source. Its exported text CSV bytes match the source-executable journey.
  Full report/reproducers are retained in `/tmp/t172-final-qa.at7ChW/`.
  After the wording correction, the same QA verified the exact delta, reran
  16 affected metadata/identity/context/planning tests and accepted the corrected
  snapshot. It added zero journey calls; the original 157 remain applicable
  because runtime, inputs and artifact are unchanged. Corrected QA archive
  SHA-256: `3316ca8c523ae8ebe6495f912eae0664ec48d095efdc76cec8566ac3be9f60d9`.
- The first full exact chain passed 3,212 tests, 7 skipped, one existing
  serializer warning (461.68s). The pre-prose-fix final exact chain also passed
  (480.30s). The post-fix exact chain and full mise gate results follow below;
  neither independent review nor CLI QA substitutes for those gates. Native
  Desktop acceptance remains pending T-173; AnkiMobile remains unverified.

## Final gates

- Post-fix exact chain passed in order: `uv run ruff check`, `uv run mypy`,
  `uv run pytest -v`: 3,212 passed, 7 skipped, one existing Pydantic serializer
  warning, 447.76 seconds. Full `mise run check` also passed, including format,
  policy and synthetic package/publish guards (3,212 passed, 7 skipped).
- `UV_OFFLINE=1 bash scripts/check-dist.sh` passed. Both wheel and sdist passed
  `bash scripts/check-install.sh "$(uv python find 3.14)"` outside the checkout
  on Python 3.14.7, with optional NLP distributions absent. Dependency install
  used the normal network policy; this is not an offline dependency-install
  claim. Distribution and install logs remain in `/tmp/t172-evidence/`.
- Before verification, Taskrail validation passed and status remained pinned
  to v0.4.0 with exactly this active task and blocked T-173 open. Off-spec
  T-128/T-179 remain untouched. Remote main still matched the accepted base.
  Runtime source, specs and T-154 goldens remain unchanged. No native acceptance,
  release publication or successor work is included.

## Implementation Notes

- Requeued behind T-180; current-base lifecycle already records this task as
  todo. Verification notes transferred from the retained T-172 checkout;
  historical STATE and lifecycle timestamps were not applied.
- 2026-10-31T10:29:52Z: verification pass
