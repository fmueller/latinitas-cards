---
id: T-113-define-annotation-sidecar-records-and-the
title: Define annotation sidecar records and the annotations segment writer
status: todo
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies:
    - T-106-segment-sentences-and-tokenize-passages-with-the
    - T-105-validate-normalized-corpora-with-corpus-validate
    - T-103-define-the-cli-result-envelope-exit-codes
updated_at: "2026-10-14T05:42:59Z"
---

# T-113-define-annotation-sidecar-records-and-the Define annotation sidecar records and the annotations segment writer

## Description

Define the `latinitas-annotations` sidecar of `specs/v0.3.0.md#sidecar-annotation-model`
and its only built-in producer, `annotations segment CORPUS --output SIDECAR [--timestamp
T] [--json]`. It implements Fixed Contracts **Identity and hashing** (`annotation_id`,
`segmentation_id`, producer kinds, anchors), **Versioned documents**, **Runtime rules**
(`created`, `timestamp_required`), **Files** (default `<corpus-stem>.annotations.jsonl`)
and the writer rules (`busy`, atomic replace, alias rejection). It consumes the corpus
reader (T-104/T-105), the `latin-seg/1`/`latin-tok/1` segmenter (T-106) and the envelope
plus `atomic_io` (T-103). Not in this task: anchor states and `annotations inspect`
(T-114), claim decisions (T-115), patches, plans and apply (T-117, T-118), policy (T-116).

## Acceptance

- `src/latinitas_cards/annotation_sidecar.py` defines strict Pydantic models: header line
  exactly `{"format":"latinitas-annotations","record":"header","version":1}`; annotation
  records carry `record: "annotation"` (`record` is part of the ID digest),
  `annotation_id`, `layer` (`segmentation|morphology|translation|correction`), anchor
  `{passage_id, passage_hash, segmentation_id?, span?: [start, end)}`, `value`, optional
  `alternatives`, `producer {kind: rule|analyzer|agent|llm|human, name, version}`
  (`version` a non-empty string, `"1"` for latin-seg; `name` matches `^[A-Za-z0-9._-]+$`),
  `input_hash` (must equal `anchor.passage_hash`), optional integer `confidence` 0..1000,
  `created` (RFC 3339 UTC) and `plan` (a `plan_id` or null). A supersede is an annotation
  record with optional `supersedes: <annotation_id>`; a retraction is `{record:
  "retraction", annotation_id, retracts, layer, anchor, input_hash, producer, created,
  plan}` where `layer`, `anchor` and `input_hash` are copied from the retracted record
  (both shapes enter the ID digest). Absent optional keys (`segmentation_id`, `span`,
  `alternatives`, `confidence`, `supersedes`) are omitted, never `null`; `plan` is always
  present (`null` outside apply). Anchor `span` = code-point offsets `[start, end)` into
  passage `text`; with `segmentation_id` both ends must fall on token boundaries of that
  segmentation. Tests reject a float anywhere (`confidence: 500.0`, a float inside
  `value`), `confidence: 1001`, layer `syntax`, producer kind `tool`, `"version": 1.0`, an
  unknown key, `input_hash` differing from the anchor and an `annotation_id` that does not
  recompute.
- `annotation_id` is `sha256:` plus the hex digest of canonical JSON of the record without
  `annotation_id`, `created` and `plan`. A golden test pins the hex for a committed record
  in `tests/fixtures/annotations/`, computed in the test with `hashlib.sha256` over a
  literal canonical JSON string (not via the code under test); changing `created` or
  `plan` keeps the ID; changing `value`, anchor or producer changes it.
- Layer values: a morphology value is an object with a required non-empty string
  `tagset` (other contents are not modelled beyond "JSON without floats"); translation
  `{language, text}` (BCP 47, non-empty text); correction `{text}` (non-empty NFC) and
  requires `span`. Segmentation `value` = `{"sentences": [{"continued": bool, "continues":
  bool, "span": [start, end], "tokens": [token_start, token_end]}], "tokenizer":
  "latin-tok/1", "tokens": [{"kind": "word"|"punct", "span": [start, end]}]}` (code-point
  offsets, half-open). The committed
  `tests/fixtures/annotations/fixture-morph.annotations.jsonl` (producer `{analyzer,
  fixture-morph, "1"}`, token spans into a segmented fixture passage) loads and
  re-serializes byte-identically; one without `tagset` is rejected.
- `annotations segment` on a committed `tests/fixtures/corpus/` corpus with `--timestamp
  2026-01-01T00:00:00Z` writes the header plus one `segmentation` record per passage in
  corpus order: producer `{rule, latin-seg, "1"}`, anchor without
  `segmentation_id`/`span`, `plan: null`, the segmentation `value` above built from
  T-106's sentences and tokens (including `continues`/`continued`); exit 0. A rerun with
  the same timestamp is byte-identical and does not rewrite the file (same inode); a fresh
  output with `--timestamp 2027-06-30T12:00:00Z` has the same `annotation_id` set;
  `SOURCE_DATE_EPOCH=0` yields `created` `1970-01-01T00:00:00Z`; with neither, exit 2
  `timestamp_required` and no file; a non-UTC or date-only `--timestamp` exits 2.
  `--timestamp` and patch `created` must match `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$`;
  `SOURCE_DATE_EPOCH` must be a non-negative integer; otherwise exit 2
  `timestamp_required` naming the bad value.
- An existing sidecar keeps its records and order; a passage gets a new segmentation
  record only if it has no current (same `passage_hash`, not superseded or retracted)
  segmentation record; a differing recomputed value is reported as a warning, not
  appended. A sidecar with another `format` or `version: 2` exits 3
  (`unsupported_format`/`unsupported_version`) with no byte change; a corpus with any
  `corpus validate` finding exits 3: the envelope carries one `invalid_corpus` error (class
  3) whose message names the first finding code, followed by the finding diagnostics (class
  1), so precedence yields 3.
- While the test holds `fcntl.flock` on `<SIDECAR>.lock`, `segment` exits 4 `busy` with no
  byte change. `--output` equal to CORPUS, a symlink or hard link to CORPUS,
  `<CORPUS>.claims.json`, `<corpus-stem>.manifest.json` or `<SIDECAR>.lock` exits 3 (code
  `alias`) with nothing written; a failure injected into `os.replace` keeps the previous
  sidecar bytes.
- `--json` emits one `CliResult` (`command: "annotations segment"`, result with appended
  and skipped counts); the test asserts `cltk`, `stanza` and `torch` are absent from
  `sys.modules` after the run. The annotation models are registered in
  `json_schemas.SCHEMAS` (`docs/schemas/latinitas-annotations.v1.schema.json`, shared drift
  test) and the matrix row `annotations segment: 0, 2, 3, 4` is added.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the pinned golden `annotation_id` and the SHA-256 of the segmented fixture
  sidecar.

## Implementation Notes

- New `annotation_sidecar.py` (named to avoid confusion with the legacy CLTK module
  `annotation.py`; models, ID digest, streaming reader, `append_records(path,
  records, protected)` built on `atomic_io.writer_lock`/`atomic_replace`/`reject_aliases`,
  and `resolve_created(flag, env)` reused by apply) and `commands/annotations.py`, a Typer
  sub-app registered as `annotations` in `_register_commands()`.
- Reuse, do not re-implement: the streaming reader `corpus_io.read_corpus` (T-104) plus
  `corpus_validation` (T-105) for the validity gate, yielding header, works and passages in
  order; `canonical.canonical_json`/`digest` (T-103); and `corpus_segmentation.segment_work`
  (T-106) mapping the ordered passages to per-passage sentences and tokens with
  `continues`/`continued`. Serialize that output as the segmentation `value` shape above.
- Lines are canonical JSON plus `\n`; appends rewrite the whole file through
  `atomic_replace`, never `open("a")`. Write `docs/annotation-sidecars.md` (record shape,
  ID rule, timestamps) and register new codes in the registry and its docs table.
- TDD order: ID golden, model rejections, segment write, rerun/timestamp, lock and alias,
  failure injection. Size guard: one domain module plus one command module.
