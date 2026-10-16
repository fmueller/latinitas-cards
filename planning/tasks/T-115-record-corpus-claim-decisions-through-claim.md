---
id: T-115-record-corpus-claim-decisions-through-claim
title: Record corpus claim decisions through claim sidecar version 2
status: completed
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies:
    - T-113-define-annotation-sidecar-records-and-the
    - T-102-capture-characterization-goldens-for-legacy
    - T-114-report-stale-orphaned-and-unmatched-annotations-in
updated_at: "2026-10-16T10:14:54Z"
---

# T-115-record-corpus-claim-decisions-through-claim Record corpus claim decisions through claim sidecar version 2

## Description

Let `claims list|record|inspect CORPUS --annotations SIDECAR` review corpus annotations
through claim sidecar version 2, per `specs/v0.3.0.md#sidecar-annotation-model` and Fixed
Contracts **Claim sidecar version 2** and **Files** (`<corpus>.claims.json`, the v0.2.1
`sidecar_path` rule). Today `claim_sidecar.Sidecar` is `version: Literal[1]`, a decision
binds only `base_fingerprint`, `Claim` requires a `DeckProfile` and `ClaimKind` lacks
`morphology`/`correction`, and `commands/claims.py` exits 1 for every failure. Deck
fingerprints, the v1 schema and its serializer must not change; the characterization
goldens (T-102) prove it. `claims` keeps its v0.2.1 JSON shape and codes; it gets no
`CliResult` envelope. Not in this task: plan/approve/apply (T-117, T-118), decision-aware
planning and apply revalidation (T-117, T-119), and the `--claims` options of
`corpus show`, `corpus validate` and `annotations inspect` (T-125).

## Acceptance

- `claim_sidecar.py` adds `CorpusBinding` (`kind: "corpus"`, `corpus_id`, `passage_id`,
  `passage_hash`, `annotation_id`, `layer`), a v2 decision (v1 `StoredDecision` fields
  plus `binding`) and `SidecarV2` (`{"version": 2, "decisions": [...]}`, no `format`).
  Every characterization v1 sidecar from T-102 loads, re-serializes byte-identically,
  and the regenerated principal-part (both recipes), form-parsing and phrase CSV exports
  equal the golden bytes.
- Deck commands reject v2: with a v2 document at `<source>.claims.json`, `claims list
  source.csv --profile profile.json --json` exits 1 with diagnostic code
  `unsupported_version`, and principal-part `preview --profile`, `generate --profile`,
  `form-parsing preview`/`export` and `phrases export`/`inspect` exit nonzero naming
  `unsupported_version` with no output file. A v1 document at `<corpus>.claims.json` is
  likewise rejected in corpus mode.
- `corpus_claim_fingerprint(record)` is unprefixed hex `sha256` of canonical JSON
  (`sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False`) of
  `{annotation_id, layer, value, anchor, producer}`; a golden test pins the hex for the
  `fixture-morph` record and shows that default `json.dumps` separators give a different
  value. Claim kinds are the four annotation layers.
- `claims list CORPUS --annotations SIDECAR --json` returns `{"claims": [...],
  "diagnostics": [...]}` with one claim per `current`, non-superseded, non-retracted
  annotation. Each corpus claim has exactly `identity` (= `passage_id`), `annotation_id`,
  `kind` (= layer), `value`, `alternatives`, `producer`, `fingerprint`, `status`
  (`accepted` iff a current accepted decision binds it, else `withheld`), `review_status`
  (open|accepted|withheld) and `reason`; a v2 decision stores `base_fingerprint` =
  `decision.claim_fingerprint` = the corpus fingerprint.
- `claims record CORPUS --annotations SIDECAR --fingerprint F --status accepted --reviewer
  R --reason X` writes version 2 with the binding copied from the annotation; `--batch`
  validates every item first and one stale or unknown item exits 1 with the file
  byte-unchanged; repeating a decision is idempotent; there is no approve-all. `--value`,
  `--evidence` or `--selected-sense` with `--annotations` exits 3 and writes nothing
  (Fixed Contracts); a batch item carrying `value`, `evidence` or `selected_sense` rejects
  the batch (exit 1, `invalid`). Combining `--annotations` with `--profile`,
  `--phrase-input`, `--selections`, `--recipes` or `--artifact` exits 1.
- Explicit exception tests retain legacy claims JSON and diagnostics, not `CliResult`:
  unreadable/malformed/v1 corpus decision files and invalid batch items fail with exit 1;
  each of the three corpus-only forbidden options fails with exit 3 and unchanged files;
  Click usage errors still exit 2. Do not generalize the contract groups' exit 3 input
  rejection to `claims`, or change deck failures from exit 1.
- Using `annotation_state` (T-114) and the committed staleness fixtures
  `tests/fixtures/corpus/wsl-three.corpus.jsonl` and `wsl-three-edited.corpus.jsonl` (line
  2 differs by one word, same `passage_id`s) and a sidecar with segmentation and
  `fixture-morph` records, accepting the line-2 morphology claim makes `claims inspect
  CORPUS --annotations SIDECAR --json` show it `accepted`; against the edited corpus the
  claim is no longer listed, `claims inspect` reports the decision `stale`, `claims
  inspect --check` exits 1, and the line 1 and 3 decisions stay accepted. A claim is
  current when its annotation's `annotation_state` is `current` and it is neither
  superseded nor retracted (so `orphaned` annotations are not claims).
- Corpus review requires a v2 decision per annotation fingerprint; an annotation with no
  decision stays `open` (no automatic acceptance) and no `--policy` option exists on
  `claims`.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the pinned corpus fingerprint and the golden comparison results.

### Execution evidence (2026-10-16)

- Base: accepted `origin/main` a4a91020612401a50e4c9df186ec87e792389f61;
  spec SHA256 fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704.
  `taskrail validate` passed; selector named T-115 before start and again before verify.
  Started 09:56:44Z; verification run 10:14:14Z passed after all review/check gates.
- V1 guard first: four sidecars load/serialize byte-identically, and all four CSV
  exports (both principal-part recipes, parsing, phrases) match committed goldens.
  `uv run pytest tests/unit/claim_goldens_test.py -q`: 4 passed before production edits.
- Initial RED: 2 failures for missing v2 models and missing deck-version diagnostic;
  then 11 expected failures for absent corpus module/options and deck error code.
  GREEN: focused corpus/golden suite passed, eventually 28 tests after review fixes.
- Independent canonical fixture bytes piped to `sha256sum` pin fixture-morph to
  `0abc30c213eaaafe24de770bc93cd0c89d5434df4d002dbfe5d12445470b01c2`;
  ordinary JSON separators yield a different digest.
- Real CLI normalize/segment/import reproduced the committed corpus/annotation bytes.
  Six claims began open; three individually selected morphology fingerprints were
  accepted. Renormalizing the edited text hid line 2, reported one stale decision
  and check exit 1; line 1/3 morphology decisions remained accepted, segmentations open.
- Temporary persistence probes: existing claims lock blocks recording, remains at
  the same inode, and is never removed; identical retry preserves exact bytes;
  injected pre-replace failure preserves prior bytes and removes staged temporary files.
- Dedicated Task loaded `code-simplifier`; accepted only separation of corpus option
  checks from report generation, removing an unused report in recording. Post-check:
  focused 25 tests, ruff and mypy passed. No rejected simplification suggestions.
- Separate parallel Tasks loaded `code-reviewer`: General used ECC code-reviewer;
  Database/persistence used database-reviewer plus postgres-patterns/database-migrations;
  Security used security-reviewer plus security-review. General and Database both:
  "No concrete task-relevant findings." These lanes cover correctness, integrity,
  locks and trust boundaries; no framework/SQL/model-provider behavior is changed.
  A separate Python lane was omitted as General plus strict mypy covered those risks.
- Fresh candidate validation confirmed the sole candidate SEC-001 below; none were
  rejected or deduplicated. Disposition: FIXED in corpus loader only, preserving
  deck behavior. At 2,000 levels Python 3.14 parsed the fixture; corrected to the
  reviewer's 100,000 levels before editing production: 3 RED failures (empty output),
  then 3 GREEN with legacy invalid diagnostics/exit 1 and unchanged persisted bytes.
- Fresh disposition-verification Task loaded code-reviewer, General/Python/Security
  guidance and python-patterns/security-review references; independently ran the
  3 regression cases and verified SEC-001 resolved, no new issues. One review/fix/
  recheck cycle; no deferred findings or newly required follow-ups.
- Final exact chain: `uv run ruff check` passed; `uv run mypy` passed (150 files);
  `uv run pytest -v`: 1,588 passed, one existing Pydantic serializer warning, 91.95s.
  `mise run check` passed: ruff, formatting, mypy, 1,588 tests (91.19s), and all
  commit/push/author/mutation policy scripts. `git diff --check` passed.
- Production scope: one new corpus_claims module and edits only to claim_sidecar
  and commands/claims. T-125 joins, T-127 exporter repair, T-128 dense-memory work
  and blocked T-038 release remain excluded; no spec or gate changes.

Validated finding, verbatim:

> FINDING SEC-001 — security
> Severity: low
> Evidence: `src/latinitas_cards/corpus_claims.py:75`; `src/latinitas_cards/commands/claims.py:101,158,188`. `load_corpus_decisions` calls `json.loads` without handling `RecursionError`, while the CLI handlers catch `OSError`, `ValueError`, and `ContractError`. Reproduced with a v2 claims file containing 100,000 nested arrays: `claims list ... --annotations ... --json` raised `RecursionError` and emitted no diagnostics.
> Finding: Deeply nested malformed corpus decision JSON escapes the legacy error handling and crashes the claims command instead of returning the required legacy diagnostic JSON.
> Failure/impact: A malformed persisted file can make `claims list`, `record`, or `inspect` fail with an uncaught exception rather than a controlled `invalid` response.
> Recommended direction: Treat `RecursionError` from decoding/validation as malformed input and test that deeply nested decision JSON returns the legacy diagnostic shape without modifying the file.

## Implementation Notes

- New `corpus_claims.py` builds corpus claims from the corpus reader `corpus_io` (T-104)
  and `annotation_sidecar.py`, behind one `current_annotations(corpus, sidecar)` function
  built on `annotation_state` (T-114). Keep
  `Claim`, `record_batch` and the principal-part checks untouched for deck mode and add a
  corpus recording path that reuses the same blocking `fcntl.LOCK_EX`,
  validate-all-then-replace protocol (factor the shared lock/replace into a helper only if
  v1 bytes stay identical). Use `<CORPUS>.claims.json.lock`, the same persistent lock
  apply will acquire nonblocking in T-118; never unlink it or introduce another lock path.
- `load_sidecar` peeks at `version` before model validation so v2 yields
  `unsupported_version` rather than a Pydantic `invalid` message; the new code is a
  `claims`/deck diagnostic, not a contract-registry entry unless a contract command raises
  it.
- Corpus mode is selected by `--annotations`; `commands/claims.py` adds that option and a
  `_corpus_state` branch beside `_state`. The `--claims SIDECAR` options on `corpus show`,
  `corpus validate` and `annotations inspect` belong to T-125.
- Update `docs/reviewed-claim-cli.md` with a corpus-mode section (version 2 shape,
  fingerprint, no authored values in corpus mode).
- TDD order: v1 golden guard first, v2 models, deck rejection, fingerprint golden, list,
  record, inspect and the stale decision. Size guard: one new module plus edits to
  `claim_sidecar.py` and `commands/claims.py`; file a follow-up for anything wider.
- 2026-10-16T10:14:14Z: verification pass
- 2026-10-16T10:14:54Z: Reviewed corpus claims v2; verification 2026-10-16T10:14:14Z passed after ruff/mypy/1588 tests, mise check and fresh disposition verification. Legacy v1 bytes/exports preserved; SEC-001 fixed; no deferred findings.
