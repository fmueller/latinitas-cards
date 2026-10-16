# Annotation sidecars

`annotations segment CORPUS [--output SIDECAR] [--timestamp T] [--json]`
appends deterministic `latin-seg/1` sentence and `latin-tok/1` token proposals.
The default output is `<corpus-stem>.annotations.jsonl`. It does not import
CLTK, Stanza or Torch, alter the corpus, or accept review decisions.

```sh
uv run latinitas-cards annotations segment tests/fixtures/corpus/valid/ordinal.jsonl \
  --output /tmp/ordinal.annotations.jsonl --timestamp 2026-01-01T00:00:00Z --json
```

The result reports `appended` and `skipped` passage counts. A new output has one
header and one segmentation per passage in corpus order. Reruns keep current
segmentations (matching passage hash, not superseded or retracted). If recomputed
content differs, `segmentation_differs` warns without appending. An ID already
anywhere in the stream is never appended again, even after retraction. Existing
records and their order are retained. A no-op does not replace the file.

## Wire format and identity

The first line is exactly
`{"format":"latinitas-annotations","record":"header","version":1}`.
All lines are canonical JSON plus a newline: sorted keys, compact separators,
UTF-8 Unicode, no floats anywhere. Unknown keys, formats and versions are rejected.
The registered schema is [latinitas-annotations.v1.schema.json](schemas/latinitas-annotations.v1.schema.json).

Annotations carry `record: "annotation"`, `annotation_id`, `layer`, `anchor`,
`value`, `producer`, `input_hash`, `created`, and `plan`. Producer kinds are
`rule|analyzer|agent|llm|human`; names use ASCII letters, digits, `.`, `_`, `-`,
and versions are non-empty strings. The segment producer is
`{"kind":"rule","name":"latin-seg","version":"1"}`.

The anchor contains `passage_id` and `passage_hash`, optionally `segmentation_id`
and `span`. `input_hash` equals `anchor.passage_hash`. Spans are non-empty
half-open code-point offsets into passage text, not UTF-8 byte offsets. With a
referenced segmentation, both ends must lie on its token boundaries.
`validate_anchor` checks bounds against a supplied passage and segmentation;
the streaming reader can validate the wire record without a corpus.

Optional `alternatives`, integer `confidence` (0–1000), `supersedes`, and optional
anchor keys are omitted when absent, never null. `plan` is required, null outside
apply. Supersession is an annotation naming `supersedes`. Retractions carry
`record: "retraction"`, `retracts`, and the shared provenance/identity fields,
but no value; layer, anchor and input hash must copy the referenced annotation.
Neither shape deletes history.

`annotation_id` is `sha256:` plus the SHA-256 of canonical JSON of the complete
record excluding only `annotation_id`, `created`, and `plan`. The `record`
discriminator is hashed. A segmentation's ID is its `segmentation_id`. Changing
timestamps or plans preserves identity; changing values, anchors or provenance
does not. The translation fixture has an independent literal canonical ID vector.

## Layer values

- `segmentation`: `{tokenizer: "latin-tok/1", tokens: [{kind: "word"|"punct",
  span: [start,end]}], sentences: [{continued: bool, continues: bool,
  span: [start,end], tokens: [token_start,token_end]}]}`. Sentences partition the
  token list; continuation across adjacent passages follows the existing segmenter.
- `morphology`: an object with a required non-empty `tagset`; other JSON contents
  are intentionally not interpreted, but cannot contain floats.
- `translation`: `{language, text}`, with a BCP 47 language and non-empty text.
- `correction`: `{text}`, non-empty NFC, with a required anchor span. It proposes
  a reading; it never changes source text. Alternatives obey the same layer shape.

Morphology, translation, supersession and retraction enter as validated records;
there is no built-in analyzer. Import/apply, claims and policy commands are not
implemented here.

## Read-only anchor inspection

`annotations inspect SIDECAR --corpus CORPUS [--check] [--json]` reports each
record's `annotation_id`, `layer`, `state`, boolean `superseded` and `retracted`,
`passage_id`, `source_sha256` (the current corpus header's source digest),
`producer`, `plan`, `created`, and `input_hash`, in sidecar order. It streams the
corpus once into an ID/hash map without retaining passage text. Text output
escapes terminal controls. Neither it nor `corpus validate --annotations SIDECAR`
nor `corpus show --annotations SIDECAR` writes files, changes mtimes, or creates locks.

Each record has exactly one anchor state, in this precedence order:

1. `unmatched`: the passage ID is absent.
2. `stale`: its passage hash differs.
3. `orphaned`: its segmentation is superseded, retracted, unknown, or is not a
   segmentation record. Unknown/retracted segmentation is the v0.3.0 rule.
4. `current`: none of the above.

Lifecycle flags are separate: a superseded passage-level segmentation can still
have a current anchor, but its token-dependent morphology is orphaned. Retraction
events are included in inspect provenance; show emits only live proposals.
No record is ever re-anchored. Inspect warnings exit 0; `--check` and validation
findings exit 1. Show hides affected proposals and exits 0. Claims/decisions,
dependency-aware planning, policy and enriched export are separate later work.

For example, after changing only `Mater filium amat.` to `Mater filiam amat.`:
before, inspect reports six current records with no diagnostics; after, the
line-2 segmentation and morphology are stale (warnings, or errors with `--check`),
and show keeps the edited text but no line-2 annotations. Other lines are unchanged.

The `annotation_states(corpus, sidecar)` API accepts a passage ID/hash mapping and
ordered annotation/retraction records. `dependents(sidecar, annotation_id)` returns
all records whose anchor names that segmentation, retaining stream order.

## Reproducible timestamps and safe writes

New records require `--timestamp` or `SOURCE_DATE_EPOCH`; no wall-clock fallback.
The explicit timestamp must be a real UTC date matching exactly
`YYYY-MM-DDTHH:MM:SSZ` (no fractional seconds or timezone offsets), and takes
precedence over the environment. `SOURCE_DATE_EPOCH` is ASCII decimal digits for
a non-negative integer representable as a UTC datetime. Missing or invalid values
exit 2 with `timestamp_required` naming the bad value. No-op reruns need no time.

Writes hold `<SIDECAR>.lock` using nonblocking `flock` from reading current
records through atomic replacement; contention exits 4 `busy`, even for no-ops.
Lock files are never removed. A same-directory staged file is fsynced and replaced
atomically, then its parent directory is fsynced. Alias outputs/locks (direct,
symlink or hard link) to corpus, claims or manifest paths are rejected before
writing. Replacement failures leave the old sidecar intact.

Any corpus validation findings reject the writer with an `invalid_corpus`
diagnostic naming the first finding, followed by the original diagnostics. Exit
precedence yields 3; the sidecar is unchanged. Unsupported sidecar formats or
versions also exit 3 without modifying it.

## Fixtures and schema regeneration

The deliberately added `annotations` target follows the existing fixture
script's preflight/`--force` and atomic-output conventions. It captures the real
segment CLI in a temporary directory and builds the required morphology record:

```sh
uv run python scripts/regen-corpus-fixtures.py --target annotations --force
```

The independent translation golden is authored, not regenerated by this target.
The `annotation-state` target captures real template normalization/segmentation,
appends token morphology via `append_records`, and adds the line-1 supersession.
It generates `wsl-three.corpus.jsonl`, `wsl-three-edited.corpus.jsonl`, full and
superseded sidecars, and the headerless `wsl-morphology.jsonl` append fragment:

```sh
uv run python scripts/regen-corpus-fixtures.py --target annotation-state --force
```

`text/wsl-two.txt` exercises removal by re-normalizing to the same corpus path;
line-3 records become unmatched. These fixed-timestamp corpora also serve T-115.
Until the existing T-127 exporter follow-up is implemented, render the new schema
through the established registry API (shared schema drift tests still apply):

```sh
uv run python - <<'PY'
from pathlib import Path
from latinitas_cards.json_schemas import SCHEMAS, render_schema
name = "latinitas-annotations"
Path(f"docs/schemas/{name}.v1.schema.json").write_text(render_schema(SCHEMAS[name]), encoding="utf-8")
PY
```
