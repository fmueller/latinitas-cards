---
id: T-107-normalize-plain-text-sources-with-manifest
title: Normalize plain-text sources with manifest-assigned and templated refs
status: todo
priority: high
spec_ref: specs/v0.3.0.md#normalized-jsonl-corpus-model
dependencies:
    - T-105-validate-normalized-corpora-with-corpus-validate
updated_at: "2026-10-14T05:42:59Z"
---

# T-107-normalize-plain-text-sources-with-manifest Normalize plain-text sources with manifest-assigned and templated refs

## Description

Add `corpus normalize SOURCE --adapter text --corpus-id ID --output FILE` with the plain-text
adapter and the `latinitas-corpus-manifest` v1 reconciler, per the plain-text paragraph of
`specs/v0.3.0.md#normalized-jsonl-corpus-model` and Fixed Contracts **Identity and hashing**
(manifest refs, `--citation-template`), **Files** and **Runtime rules**. The reconciler reuses
only the matching rules of `manifest.reconcile_csv_manifest` (unique exact fingerprint reuse,
duplicate, edited and removed review kinds, explicit reuse/allocation/removal approvals) with
`passage_hash` as fingerprint; it does not reuse `CsvIdentityManifest`, its scope or its
`csv-source-NNNNNN` allocator. The USFX adapter is T-109; the `--json` envelope and codes come
from T-103; validation reuses T-105.

## Acceptance

- Options exactly: `corpus normalize SOURCE --adapter usfx|text --corpus-id ID --output
  FILE [--language TAG] [--descriptor FILE] [--manifest FILE] [--citation-template T]
  [--approve-reuse N=REF]... [--approve-allocation N]... [--approve-removal REF]...
  [--json]` (`--adapter usfx` exits 2 `usage_error` "not yet supported" until T-109).
  `--language` must match `^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$`, else exit 2
  `usage_error`. One passage per line whose canonical text is non-empty; `order` 1-based
  in document order; `N` = `order - 1`.
- First run with no manifest allocates `p000001`... for every passage (duplicates included)
  in document order, writes `<output-stem>.manifest.json` (or `--manifest FILE`) and then the
  corpus, exit 0, no approval flags; citations use scheme `ordinal`, parts `[{"unit":
  "passage", "value": "p000001"}]`, and the output passes `corpus validate`. A rerun on
  unchanged input (same ordered hash list as the manifest snapshot) reuses every ref
  positionally: `tests/fixtures/corpus/text/refrain.txt` (repeated lines) reruns with exit 0
  and byte-identical corpus and manifest.
- After an unrelated insertion, unique exact-hash matches keep their refs and the inserted
  line is an `inserted` review item; an edited line is `edited` (naming the candidate ref), a
  new duplicate `duplicate`, a missing line `removed`. Any unresolved item exits 1 with
  `review_required` diagnostics (`item` = N or null, `ref` = candidate ref) and writes
  nothing (corpus and manifest bytes unchanged). `--approve-reuse N=REF`,
  `--approve-allocation N` (next never-used ref) and `--approve-removal REF` (retires it)
  resolve them; the manifest is replaced before the corpus. An existing `--output` (and
  manifest) is replaced atomically, never refused with `exists`; identical bytes leave the
  file untouched (same inode). `write_artifact` is for plans and receipts only.
  `result.review_items` lists `{kind: edited|inserted|duplicate|removed, ordinal: N|null,
  candidates: [ref...]}` in ordinal order, removed items last; `inserted` is this format's
  name for the CSV reconciler's `unmatched`.
- Manifest-backed writes are recoverable two-file commits, not atomic visibility across
  paths. Hold `<output>.lock` and `<manifest>.lock` from old-state reads through both
  replacements, in lexical resolved-path order, nonblocking (exit 4 `busy`), never
  unlinking locks. Validate reconciliation and stage/fsync both files before replacing
  the manifest, then the corpus; fsync each parent directory after its replacement.
- Required manifest `materialization` stores `request_digest`, `corpus_digest` and
  `previous_corpus_digest` (null on first creation), retained after success as the last
  committed transition receipt. Request digest covers source raw digest and basename,
  corpus ID, adapter/rule versions, language, descriptor content and canonical approvals
  (reuse pairs sorted by ordinal/ref, allocation ordinals/removal refs sorted); reject
  duplicate/contradictory flags before committing. Never hash paths or timestamps.
  Corpus digests cover file bytes; do not reconcile/advance until the corpus matches
  the committed target digest. No third journal or new recovery command.
- If the corpus is absent or matches `previous_corpus_digest`, an identical request
  rebuilds using committed assignments, verifies `corpus_digest`, and replaces only the
  corpus without reapplying approvals or advancing `next_ref`. A different request,
  unexpected corpus bytes or irreproducible target digest exits 4 `binding_drift` with
  both files unchanged. When corpus bytes already match the target, an identical request
  (including original approval flags) is a byte-identical no-op; a new request reconciles.
- Adversarial recovery tests inject failure before manifest replacement, between the
  two replacements, and after corpus replacement. The identical approved rerun must
  equal an uninterrupted run byte for byte, including duplicates, removed/retired refs
  and `next_ref`; no double allocation or approval consumption. Cover first creation
  with absent old corpus, same-target no-op and a subsequent distinct valid request.
  Pending recovery rejects independently changed source, basename, descriptor, corpus ID,
  language or approvals, unexpected corpus bytes and failed digest reproduction (4,
  unchanged files). Reordered equivalent flags and relocated paths keep request identity.
  Expected replacement failures exit 3 `unreadable`; after manifest commit instruct the
  caller to rerun the identical request. Clean temps on handled errors; interrupted-run
  orphan temps are never committed state. Read-only consumers need only the complete
  old/new corpus, never the manifest.
- `--approve-reuse` naming a retired ref exits 3 (`retired_ref`); an unknown ref, an N out
  of range, conflicting approvals, or approval flags on a run without an existing manifest
  exit 3 (`invalid_approval`); a retired ref is never
  reassigned by allocation (test retires `p000002`, allocates, asserts `p000004` not `p000002`).
- `--citation-template wsl:work.section.line` reads `LABEL<TAB>text` lines from
  `tests/fixtures/corpus/text/wsl.txt`: the label is the `native_ref` (components split on
  `.`, count must equal the units, each encoded with `encode_ref_component`), parts carry
  the raw component strings, a `work` unit creates `work` records in first-appearance
  order, without one the single work is `--corpus-id`; no manifest is read or written. A
  malformed template exits 2 (`usage_error`); a line without a tab or with a wrong
  component count or an empty component exits 3 (`invalid_line`); a duplicate label exits 3
  (`duplicate_source_ref`). `--manifest` or any approval flag with `--citation-template`
  exits 2. Test raw `%`, `%2E` and non-ASCII label components: encode once from NFC
  (`%25`, `%252E`, `%C3%A9`), preserve raw citation values and validate encoded refs.
- `--descriptor FILE` is accepted and parsed with the T-104 descriptor model; header and
  `corpus` descriptor fields are null without it (T-109 tests the descriptor values).
  Header `source` is `{file_name: basename, sha256: "sha256:<hex of raw bytes>", bytes}`.
  Reject direct, symlink and hard-link aliases among both outputs, SOURCE, descriptor
  and both lock paths before writing, allowing intended replacement of each output
  itself. Test each output against each protected class, including output/manifest
  aliasing. A held corpus or manifest lock exits 4 `busy`; two output paths sharing a
  manifest contend on the same manifest lock with no file changes. Matrix row `corpus
  normalize: 0, 1, 2, 3, 4` (2 = `usage_error` for `--manifest`/approval flags with
  `--citation-template` or `--adapter usfx`, a malformed template or `--language`).
- Determinism: two subprocess runs (`sys.executable -m latinitas_cards corpus normalize ...`)
  with different cwd, `TZ`, `LC_ALL` and `PYTHONHASHSEED` produce byte-identical corpus and
  manifest, containing no absolute path, hostname or clock value.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- Modules: `src/latinitas_cards/corpus_text.py` (line reading, template parsing, record
  building) and `src/latinitas_cards/corpus_manifest.py` (manifest model + pure reconciler
  returning assignments and review items); the command lives in `commands/corpus.py`.
- Manifest v1 shape (Pydantic strict, registered in `json_schemas.SCHEMAS` as
  `docs/schemas/latinitas-corpus-manifest.v1.schema.json`): `format`,
  `version`, `corpus_id`, `next_ref` (int), `entries: [{ref, passage_hash, ordinal, active}]`,
  `snapshot: [passage_hash...]` in order, and required `materialization` as above.
  Persist ordered assignments, retired refs and allocator state. No scope or timestamps.
- Read SOURCE as UTF-8 (strip one leading BOM; undecodable bytes exit 3 `unreadable`);
  split lines on `\n`/`\r\n`; canonicalize with `corpus_model.canonical_text`.
- Reuse T-103's ordered multi-lock, alias and staged-replacement primitives. Extend
  `corpus_io` only as needed to serialize/stage a corpus under already-held locks;
  do not reacquire the corpus lock or independently commit it before staging the manifest.
  Keep request digest, recovery and transition orchestration in `corpus_manifest.py`.
- Register every new code in the T-103 registry and docs table.
- Create `docs/normalized-corpus.md` (record shapes, refs, manifest flags,
  `--citation-template`); T-109 and T-110 add the USFX section.
- TDD order: manifest/reconciliation vectors, interrupted transition recovery, locks and
  aliases, then command/template wiring. Size guard: the two modules above, necessary
  `corpus_io`/shared I/O integration, schemas, fixtures/tests and `corpus normalize`.
  These recovery contracts are required here, not deferred to a follow-up by size guards.
