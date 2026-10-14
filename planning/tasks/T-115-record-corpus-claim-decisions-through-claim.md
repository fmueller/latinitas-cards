---
id: T-115-record-corpus-claim-decisions-through-claim
title: Record corpus claim decisions through claim sidecar version 2
status: todo
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies:
    - T-113-define-annotation-sidecar-records-and-the
    - T-102-capture-characterization-goldens-for-legacy
    - T-114-report-stale-orphaned-and-unmatched-annotations-in
updated_at: "2026-10-14T05:42:59Z"
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
routing (T-119), and the `--claims` options of `corpus show`, `corpus validate` and
`annotations inspect` (T-125).

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

## Implementation Notes

- New `corpus_claims.py` builds corpus claims from the corpus reader `corpus_io` (T-104)
  and `annotation_sidecar.py`, behind one `current_annotations(corpus, sidecar)` function
  built on `annotation_state` (T-114). Keep
  `Claim`, `record_batch` and the principal-part checks untouched for deck mode and add a
  corpus recording path that reuses the same blocking `fcntl.LOCK_EX`,
  validate-all-then-replace protocol (factor the shared lock/replace into a helper only if
  v1 bytes stay identical).
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
