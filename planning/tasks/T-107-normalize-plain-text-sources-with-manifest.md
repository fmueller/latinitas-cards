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
  component count exits 3 (`invalid_line`); a duplicate label exits 3
  (`duplicate_source_ref`). `--manifest` or any approval flag with `--citation-template`
  exits 2.
- `--descriptor FILE` is accepted and parsed with the T-104 descriptor model; header and
  `corpus` descriptor fields are null without it (T-109 tests the descriptor values).
  Header `source` is `{file_name: basename, sha256: "sha256:<hex of raw bytes>", bytes}`.
  Outputs aliasing SOURCE, the manifest, the descriptor or a `.lock` (direct, symlink, hard
  link) exit 3 (`alias`); a held `<output>.lock` exits 4 (`busy`). Matrix row `corpus
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
  `snapshot: [passage_hash...]` in order. No scope, no timestamps.
- Read SOURCE as UTF-8 (strip one leading BOM; undecodable bytes exit 3 `unreadable`);
  split lines on `\n`/`\r\n`; canonicalize with `corpus_model.canonical_text`.
- Write the corpus through `corpus_io.write_corpus` and the manifest through
  `atomic_io.atomic_replace` under the same lock (alias check, fsync, replace).
- Register every new code in the T-103 registry and docs table.
- Create `docs/normalized-corpus.md` (record shapes, refs, manifest flags,
  `--citation-template`); T-109 and T-110 add the USFX section.
- Size guard: the two modules above plus `corpus normalize`. File a follow-up rather than
  widen.
