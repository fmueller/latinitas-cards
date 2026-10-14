---
id: T-114-report-stale-orphaned-and-unmatched-annotations-in
title: Report stale, orphaned and unmatched annotations in inspect and validate
status: todo
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies:
    - T-113-define-annotation-sidecar-records-and-the
    - T-107-normalize-plain-text-sources-with-manifest
updated_at: "2026-10-14T05:42:59Z"
---

# T-114-report-stale-orphaned-and-unmatched-annotations-in Report stale, orphaned and unmatched annotations in inspect and validate

## Description

Report anchor states from `specs/v0.3.0.md#sidecar-annotation-model` and Fixed Contracts
**Identity and hashing** ("hash mismatch is `stale`, superseded segmentation `orphaned`,
missing passage `unmatched`") through `annotations inspect SIDECAR --corpus CORPUS
[--check] [--json]` and `corpus validate CORPUS --annotations SIDECAR`, and omit such
annotations from `corpus show CORPUS [--passage ID]... [--limit N] [--annotations
SIDECAR]`. Nothing is ever re-anchored. It builds on the sidecar model (T-113) and uses
plain-text fixtures normalized with `--citation-template` (T-107). No other task adds `corpus
show`; this task adds it to `commands/corpus.py` (from T-105). Not in this task: `--claims`
on any command (T-125), decision staleness (T-115), dependency-aware plan routing (T-117).

## Acceptance

- `src/latinitas_cards/annotation_state.py` exposes `annotation_states(corpus, sidecar)`
  returning, per `annotation_id`, an anchor state `current|stale|orphaned|unmatched` and
  lifecycle flags `superseded`/`retracted`, plus `dependents(sidecar, annotation_id)`
  (records whose anchor `segmentation_id` names it). `unmatched`: `passage_id` absent from
  the corpus; `stale`: anchor `passage_hash` differs from the corpus passage; `orphaned`:
  `segmentation_id` names a segmentation record that a later record supersedes. Unit tests
  cover each state and an unchanged control.
- Fixtures: `tests/fixtures/corpus/text/wsl-three.txt` (three `LABEL<TAB>text` lines),
  `text/wsl-three-edited.txt` (one word changed in line 2) and `text/wsl-two.txt` (line 3
  removed), normalized with `corpus normalize ... --adapter text --citation-template
  wsl:work.section.line`, and `tests/fixtures/annotations/` JSONL fragments of morphology
  records anchored to line tokens; commit their normalized corpora as
  `tests/fixtures/corpus/wsl-three.corpus.jsonl` and `wsl-three-edited.corpus.jsonl` for
  T-115. Flow: normalize, `annotations segment --timestamp 2026-01-01T00:00:00Z`, append the
  morphology records with `annotation_sidecar.append_records`,
  re-normalize the edited source to the same corpus path; `annotations inspect SIDECAR
  --corpus CORPUS --check --json` exits 1 with exactly the line-2 annotations reported
  `stale` (`ref` = `annotation_id`), lines 1 and 3 `current`; without `--check` it exits 0
  with the same findings as warnings; `corpus validate CORPUS --annotations SIDECAR
  --json` exits 1 with the same codes. The unedited control exits 0 with no diagnostics in
  both commands.
- Re-normalizing `wsl-two.txt` reports line 3 annotations `unmatched` (exit 1 with
  `--check`). A committed sidecar where a later record supersedes line 1's segmentation
  reports line 1's token-anchored morphology `orphaned` while line 1's passage-level
  records and all line 2 and 3 records stay `current`.
- `annotations inspect --json` lists for every record its `annotation_id`, layer, state,
  flags, `passage_id`, the corpus header `source.sha256`, producer, `plan` and `created`,
  plus `input_hash`, in sidecar order; text output escapes controls. This is the
  provenance surface consumed by T-119; it requires no claim join.
- `corpus show CORPUS --annotations SIDECAR --json` returns passages in corpus order
  (default `--limit 20`, `--limit 1000` accepted; `--limit` is a Click `IntRange(1, 1000)`,
  so `--limit 1001` and `--limit 0` exit 2; repeated `--passage ID` selects those IDs and an
  unknown ID exits 3 `unknown_passage`) with only `current`, non-superseded, non-retracted
  annotations: after the edit flow the line-2 annotations are absent and lines 1 and 3
  unchanged. `result = {"passages": [{"passage_id", "citation", "text", "annotations":
  [{"annotation_id", "layer", "value", "producer"}]}], "truncated": bool}`.
- Read-only: `annotations inspect`, `corpus validate` and `corpus show` leave corpus and
  sidecar bytes unchanged and create no `.lock` file; an unsupported or invalid sidecar
  exits 3 (`unsupported_format`, `unsupported_version`, `invalid_sidecar`). Matrix rows:
  `annotations inspect: 0, 1, 3`, `corpus show: 0, 3`, and `corpus validate` gains its
  `--annotations` triggers.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the `--check` JSON of the edit, removal and supersede fixtures.

## Implementation Notes

- New `annotation_state.py` for the state logic; `annotations inspect` goes in
  `commands/annotations.py`; `corpus validate --annotations` and `corpus show` extend
  `commands/corpus.py` (from T-105) through a thin callback that calls `annotation_state`.
  Use `run_contract` and registry codes `stale`, `orphaned`, `unmatched`.
- A record has one anchor state, checked in the order `unmatched`, `stale`, `orphaned`. An
  unknown `segmentation_id` and a retracted segmentation are not fixed by the spec: treat
  both as `orphaned`. This is the v0.3.0 rule; document it in `docs/annotation-sidecars.md`.
- Streaming: read the corpus once into a `passage_id -> passage_hash` map; never hold
  passage text for `inspect`.
- `corpus show` is a bounded read-only view (no export format, no `--output`), per
  `#sidecar-annotation-model`; v0.6.0 owns enriched export.
- TDD order: state unit tests, inspect, validate, show. Size guard: one module plus
  callbacks in two existing command modules.
