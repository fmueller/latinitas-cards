---
id: T-104-define-versioned-corpus-records-with-a-canonical
title: Define versioned corpus records with a canonical JSONL reader, writer and schemas
status: todo
priority: high
spec_ref: specs/v0.3.0.md#normalized-jsonl-corpus-model
dependencies:
    - T-103-define-the-cli-result-envelope-exit-codes
updated_at: "2026-10-14T05:42:59Z"
---

# T-104-define-versioned-corpus-records-with-a-canonical Define versioned corpus records with a canonical JSONL reader, writer and schemas

## Description

Define the `latinitas-corpus` v1 record models and the shared canonical primitives every
later corpus task builds on, per `specs/v0.3.0.md#normalized-jsonl-corpus-model` and the Fixed
Contracts items **Identity and hashing**, **Versioned documents** and **Corpus records**.
Nothing corpus-shaped exists today; reusable precedent is the canonical JSON in
`identity._note_identity_payload` and the strict-version models in `claim_sidecar.py`. This task
ships models, hashing, `latin-fold/1`, native_ref/passage_id rules, a streaming JSONL reader, a
canonical writer built on T-103's locked atomic replace primitives, and committed JSON Schemas.
It adds no command: `corpus validate` is T-105, adapters are T-107 and T-109, segmentation is
T-106.

## Acceptance

- Pydantic v2 strict models in `src/latinitas_cards/corpus_model.py`: header (`format:
  "latinitas-corpus"`, `version: 1`, `record: "header"`, `corpus_id`, `language` default `la`,
  `text_canon: 1`, `norm_rules`, `adapter` in `usfx/1|text/1`, `source: {file_name, sha256,
  bytes}`, `edition`, `publisher`, `url`, `license` nullable), `corpus` (`corpus_id`, `title`,
  `period`, `register` nullable free strings), `work` (`work`, `title`, `order`), `passage`
  (`passage_id`, `work`, `native_ref`, `order`, `citation`, `text`, `norm`, `passage_hash`,
  optional `translations: [{language, text, provenance}]`, `notes`, `headings`,
  `source_markup`, `source_extra`), plus `latinitas-corpus-descriptor` v1. `extra="forbid"`;
  `version: 1.0`, `true` or `"1"` is rejected; citation `parts[].value` must be `str`.
  `norm_rules: "latin-fold/1"`; `work.title` nullable; `work.order` and `passage.order` are
  1-based, contiguous, passage order global across works; descriptor v1 fields `edition,
  publisher, url, license, title, period, register` (all optional strings).
  `citation.display`: `usfx-bcv` `"<book> <chapter>:<verse>"`; `ordinal` the ref; templates
  the raw label.
- `corpus_id` regex `^[a-z0-9][a-z0-9-]{1,62}$`; `encode_ref_component()` percent-encodes
  UTF-8 bytes outside `[A-Za-z0-9_-]` with uppercase hex (`.` -> `%2E`, `3a` unchanged);
  `passage_id = corpus_id + ":" + native_ref`, NFC, at most 256 UTF-8 bytes; tests cover
  each rejection.
- `canonical_text()` = NFC, every Unicode whitespace run (incl. U+00A0, tab, newline) to one
  U+0020, trimmed; `passage_hash()` = `"sha256:" + sha256(utf8(text)).hexdigest()`.
  `tests/fixtures/corpus/schema/hash-vectors.json` pins three passages (Vulgate GEN 1:1
  `In principio creavit Deus cælum et terram.`, an NFD input with combining macrons, and a
  whitespace-run input with `" : "`); the test recomputes each expected hash independently with
  `hashlib` from the expected canonical text literal, and NFD input hashes equal NFC.
- `latin_fold()` (`latin-fold/1`): NFC, NFD, strip combining marks, casefold, `æ`->`ae`,
  `œ`->`oe`, `j`->`i`, `v` unchanged; parametrized cases cover precomposed and combining
  macrons and breves, `Æ`/`æ`, `Œ`/`œ`, `J`/`j` and `v`/`V`. `norm` never enters an ID or hash.
- `corpus_model` uses `canonical.canonical_json`/`digest` from T-103 (no second
  implementation); tests pin the float rejection and the `sha256:` prefix for corpus
  records. Writer emits one canonical record per line ending `\n`; header first. A
  round-trip test reads and rewrites `tests/fixtures/corpus/valid/{usfx-bcv,ordinal,wsl}.jsonl`
  (schemes `usfx-bcv`, `ordinal` and a work/section/line template `wsl`) to identical bytes.
  The header line of `valid/usfx-bcv.jsonl` and one passage line are also asserted against
  literal strings in the test.
- Reader streams line by line and raises typed errors the CLI maps to exit 3: unreadable
  file, non-JSON or non-object line, missing header, unknown `format`, `version` > 1 or
  non-integer; JSON floats are detected via `parse_float` and reported, not coerced. The
  gate checks only: line 1 parses as a JSON object with `record: "header"`, `format:
  "latinitas-corpus"` and a strict integer `version` <= 1. All other header and record model
  violations (including `corpus_id`) are yielded as findings, not raised.
- Both models are registered in `json_schemas.SCHEMAS` (T-103); the shared drift test
  covers `docs/schemas/latinitas-corpus.v1.schema.json` (union discriminated on `record`)
  and `docs/schemas/latinitas-corpus-descriptor.v1.schema.json`.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the three pinned hash values in the task note.

## Implementation Notes

- Modules: `corpus_model.py` (models, canonical text, hashing, fold, ref rules) and
  `corpus_io.py` (streaming reader `read_corpus(path)` yielding typed records and
  `write_corpus(path, records, protected)`, which serializes with `canonical_json` and
  writes only through T-103's `atomic_io.writer_lock`, `reject_aliases` and
  `atomic_replace`; no second lock, alias or `Busy` implementation). Alias kinds and failure
  injection are tested in T-103; here test one wiring case each (`busy`, `alias`).
- Create `scripts/regen-corpus-fixtures.py --target <name>` (deliberate updates only, never
  called by tests; refuses to overwrite an existing file unless `--force`). It must never
  write under `tests/fixtures/goldens/legacy-vulgate/` or `claims-v1/` (T-102's
  `scripts/regen-legacy-goldens.py` owns those). The three valid fixtures are written once
  by `scripts/regen-corpus-fixtures.py --target valid` using the model API and
  committed; tests never regenerate them.
- TDD: hash vectors and fold cases first (RED), then models, then reader/writer.
- Size guard: two modules and one regen script, no command. File a follow-up rather than
  widen.
