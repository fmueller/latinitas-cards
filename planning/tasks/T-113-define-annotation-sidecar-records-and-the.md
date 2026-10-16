---
id: T-113-define-annotation-sidecar-records-and-the
title: Define annotation sidecar records and the annotations segment writer
status: completed
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies:
    - T-106-segment-sentences-and-tokenize-passages-with-the
    - T-105-validate-normalized-corpora-with-corpus-validate
    - T-103-define-the-cli-result-envelope-exit-codes
updated_at: "2026-10-16T09:20:55Z"
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

## Workflow-v3 Evidence

| Step | Evidence | Result |
| --- | --- | --- |
| Understand | Fetched origin; HEAD and origin/main matched accepted base 3b4e8ff99978de1d78ab9dc613f561752781b667. Spec SHA-256 fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704 matched. Taskrail validate passed; next selected T-113, off_spec false; start recorded 2026-10-16T08:54:09Z. | Exact authoritative task, no conflicting owner/spec drift. |
| Strict TDD | Literal ID test first failed missing module, then passed. Rejection/shape tests: 10 failed, 19 passed before implementation, then 29 passed. Real segment tests: 12 failed missing command, then 41 combined passed. Lock/alias/failure phase: 9 failed, 17 passed, then 55 combined passed. | ID, rejection, writer/time, then persistence sequence retained. |
| Edge-case TDD | Contextual spans and layered corpus gate tests failed before fixes. Retraction binding test failed before reader/writer validation. Historical retracted duplicate ID test failed timestamp_required before moving timestamp resolution after ID deduplication. Invalid explicit timestamp on no-op failed before unconditional explicit-flag validation. | No clock needed for actual no-op; bad explicit flags still rejected. |
| Initial checks | ruff check; mypy; pytest -v. First full suite exposed the existing registration test's missing annotations-group traversal (1 failed, 1536 passed); extended the test using its established pattern. Repeated full chain: 1539 passed, one existing serializer warning. | Green before simplification/review. |
| Simplify | Dedicated Task loaded personal code-simplifier; indexed current segmentations by passage rather than scanning full history per passage. ruff/mypy and 139 focused tests passed; inspected diff. | Behavior and contracts unchanged; no rejected simplifications. |
| Review | Separate parallel read-only Tasks: General, Security, Database (filesystem persistence/concurrency), Python. Each instructed to load personal code-reviewer and reviewer-index ECC guidance. General: code-reviewer/common rules; Security: security-reviewer/security-review/cloud support; Database: database-reviewer/postgres-patterns/database-migrations (SQL-specific portions not applicable); Python: python-reviewer/python-patterns. | General plus three specialists within budget; no other framework/domain lanes apply. |
| Candidate validation | Fresh read-only code-reviewer candidate-validation Task reproduced General/Security candidates, rejected false Database runtime assertion, and deduplicated its schema concern into SEC-001. Python: "No concrete task-relevant findings." | Two validated findings; no invented findings. |
| Disposition/recheck | Both validated findings fixed with six red regressions, then six green; 151 focused tests passed. Fresh code-reviewer disposition verifier independently reran 151 focused tests: both resolved, Database candidate rejected, no new in-scope issues. | One risky review/fix cycle; no remaining findings. |
| Final gates | 2026-10-16T09:19Z: exact ruff check / mypy / pytest -v chain passed (1545 tests, one pre-existing corpus serializer warning); mise run check passed (1545 tests, format check and all shell policy/mutation suites). git diff --check passed. | No suppressed failures or altered gates. |

### Independent review candidates and dispositions

**T113-G-001, correctness, medium — fixed.** Verbatim finding:

> read_sidecar accepts non-canonical JSONL records even though the sidecar format requires canonical lines.

Reviewer evidence: annotation_sidecar.py's json.loads/record validation accepted
reordered, whitespace-padded input; a no-op preserved the noncanonical stream.
Impact: accepted sidecars did not meet the documented wire contract.
Recommended direction: compare each line with canonical_json(raw) plus newline.
New reader cases for spaces, reordered keys, no final newline, duplicate keys,
and CRLF all failed "DID NOT RAISE ContractError" before the fix. They pass with
exact canonical comparison and newline translation disabled. Format/version gates
still precede the canonical check. Canonical fixture bytes and IDs are unchanged.

**SEC-001, security, low — fixed.** Verbatim finding:

> The published annotation schema accepts an incomplete header that the sidecar contract and runtime reader reject.

Reviewer evidence: defaulted header fields left the generated AnnotationHeader
definition without required keys; an empty object satisfied that schema branch.
Impact: schema consumers and the wire reader could disagree about valid input.
Recommended direction: require the header fields and add missing-key regressions.
The schema/model test failed required=set() before removing defaults; it now
requires format, record and version, and each missing key is rejected. The writer
constructs all fields explicitly. Regenerated through render_schema, not the
excluded exporter. A temporary jsonschema Draft202012Validator independently
accepted the exact header, rejected the empty/missing-key variants, and validated
all committed annotation fixture lines. No dependency changes were retained.

**T113-DB-001, correctness, medium — rejected.** Verbatim candidate:

> The sidecar reader accepts a header missing "record": "header", despite the task requiring the first line to be exactly the specified three-key object.

Fresh candidate validation reproduced the contrary result: the discriminated
union raises ContractError for the missing record tag, so the claimed runtime
acceptance/silent normalization did not occur. Its schema concern duplicates
SEC-001. Fresh disposition verification confirmed the rejection. No false
runtime fix was applied; no findings were deferred.

### Real workflow and fixture evidence

- Manual plan/report executed in a TemporaryDirectory through real CLI
  subprocesses: corpus validate exit 0; segment appended 3/skipped 0; timestamp-free
  rerun appended 0/skipped 3 with identical bytes and inode 1572133; different
  timestamp retained the ordered ID set. Missing/date-only timestamp exited 2
  without output; epoch zero yielded exactly 1970-01-01T00:00:00Z.
- Invalid corpus emitted invalid_corpus then hash_mismatch (exit 3); unsupported
  version and alias exited 3 without byte changes; parent-held flock caused the
  separate CLI process to exit 4 busy. Replacement injection unit regression kept
  previous bytes and removed staged temporary files.
- Fixture morphology anchor [0,5] selected Māter using Unicode code points and
  referenced token boundaries. All fixture records roundtrip byte-identically.
- Independent literal golden annotation_id:
  sha256:2ee246345cc09c45bba19839e626dad38091beae1a34698cc9df7d2bd98a634b.
- Segmented ordinal fixture SHA-256:
  cb822ae209d8905721a582b3e271582f8da76f9226a1c017ccf167afb617ce4e.
- The missing annotations fixture-generator target was added minimally using the
  existing preflight/--force/atomic-output pattern; it captures the real CLI in a
  temporary directory. Forced regeneration was byte-identical; non-force rerun
  refused overwrite with exit 2. No unrelated generator helpers were introduced.
- Size guard: one new domain module and one command module. Legacy/goldens and
  pinned spec are unchanged. No inspect/policy/claims/plans/apply functionality,
  no CLTK/Stanza/Torch imports, no release/tag/deployment/publication, and no gate,
  priority or dependency changes. T-127 exporter follow-up remains untouched;
  T-128 future dense-metadata hardening remains unstarted and excluded. No new
  follow-up tasks were needed.
- 2026-10-16T09:20:55Z: verification pass
- 2026-10-16T09:20:55Z: Implemented annotation sidecars and segment producer after all workflow-v3 reviews, fresh disposition verification and full gates; see task evidence.
