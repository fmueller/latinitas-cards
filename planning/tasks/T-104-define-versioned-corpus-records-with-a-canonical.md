---
id: T-104-define-versioned-corpus-records-with-a-canonical
title: Define versioned corpus records with a canonical JSONL reader, writer and schemas
status: completed
priority: high
spec_ref: specs/v0.3.0.md#normalized-jsonl-corpus-model
dependencies:
    - T-103-define-the-cli-result-envelope-exit-codes
updated_at: "2026-10-15T22:27:53Z"
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
  NFC UTF-8 bytes once outside `[A-Za-z0-9_-]` with uppercase hex (`.` -> `%2E`,
  `%` -> `%25`, `é` -> `%C3%A9`, `3a` unchanged); join components with literal `.`.
  `native_ref` matches `^(?:[A-Za-z0-9._-]|%[0-9A-F]{2})+$`; validators check the
  encoded spelling without decoding/re-encoding. Tests accept `GEN.1.1%2E2`, `%25` and
  `%C3%A9`, reject bare `%`, `%2`, `%GG`, lowercase `%2e`, raw non-ASCII and empty refs,
  and distinguish raw source `%2E` (encoded `%252E`) from raw source `.` (`%2E`).
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

## Implementation and workflow evidence (2026-10-15)

- Baseline fetched and verified before start: local HEAD and origin/main both
  `f19f9e7c9735f2df2e499fe929cbfa272d3725e6`. Taskrail validate passed and
  next --json selected this exact task. Start timestamp: 2026-10-15T22:04:02Z.
- Pinned spec SHA256 before start and finalization:
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
- Scope: two new domain modules, one regeneration script, registry/drift-test
  extension, two test files, two generated schemas, corpus fixtures. No commands,
  spec changes, legacy golden changes or edits to accepted T-103 primitives.
  Corpus-wide semantic checking remains T-105; the reader preserves raw objects
  and model errors in recoverable findings for that consumer.
- RED/GREEN order: `uv run pytest -q tests/unit/corpus_model_test.py`
  first failed collection for absent corpus_model, then 7 hash/fold tests passed;
  the model stage failed collection for absent CorpusDescriptor, then 33 passed.
  `uv run pytest -q tests/unit/corpus_io_test.py` first failed collection for
  absent corpus_io. Adding I/O passed 13 cases, while fixture/schema tests failed
  for the intentionally absent committed files; generating them made the combined
  model/I/O/schema set pass 53 tests. Subsequent boundary/wire tests brought it to 57.
- Pinned passage hashes, independently recomputed with hashlib from literal
  canonical text (all with `sha256:` prefix):
  - GEN 1:1: `f231be94df2e0e4e7899e12485f82668689fe2d6596d0ab5af2b31e484619e73`.
  - Combining-macron input: `b26e700a2726398b8b49c503fceb06eeb8d4f79b1c82bebd8a9a58bbe6c55f8c`.
  - Whitespace/colon input: `190774e394bc0bd622a3865bd566dabe3349075b3f38fc96bd980aaee610d4d3`.
- Initial mandatory chain passed: Ruff; mypy 125 source files; pytest 1159 passed,
  1 skipped (57.46s). Dedicated code-simplifier Task loaded the personal skill and
  replaced per-byte regular-expression matching with an ASCII safe-byte set.
  Accepted that narrow simplification; post-simplification focused tests: 57 passed.
- Separate parallel read-only code-reviewer Tasks: General (ECC code-reviewer,
  common rules), Python (ECC python-reviewer, python-patterns), Security (ECC
  security-reviewer, security-review), Database/persistence (ECC database-reviewer,
  postgres-patterns, database-migrations). Persistence lane selected for file locks
  and atomic replacement, not SQL. No framework/ML/RAG/healthcare/network lanes:
  those domains are not affected. Three specialists plus mandatory General.
- A fresh candidate-validation Task loaded code-reviewer and validated all five
  candidates, consolidating F1 into T104-SEC-001. No initial candidates rejected.

### Verbatim validated findings and dispositions

- F-001: "The model accepts citations whose display contradicts the declared
  scheme, despite T-104 specifying `usfx-bcv` display as `\"<book> <chapter>:<verse>\"`
  and `ordinal` display as the reference." Fixed: scheme-specific part-unit and
  display validation; templates retain the raw joined label.
- F-002: "Header language values are not validated as BCP 47 tags." Fixed: shared
  well-formed RFC 5646 syntax (including grandfathered/private-use tags), duplicate
  variant/extension checks, used by headers and translations; no registry lookup.
- F-104-1: "The versioned document models and schemas accept documents missing
  required `format` and/or `version` keys." Fixed: both envelope keys required;
  generated schemas and tests updated without changing T-103's model family.
- T104-SEC-001: "The fixture regeneration script can write outside its intended
  output directories through a symlink, including into the legacy goldens or
  claims fixtures it must not touch." Fixed: preflight every output and ancestor
  for symlinks before any write, including under --force.
- F1 (deduplicated): "The fixture generator can overwrite an existing output
  symlink despite its no-overwrite contract." Covered by the same guard, with a
  dangling-symlink regression.
- Cycle-one strict-TDD command:
  `uv run pytest -q tests/unit/corpus_model_test.py tests/unit/corpus_io_test.py -k 'envelope_required or citation_display or invalid_language_tags or regenerator_refuses'`
  failed 22 cases (64 deselected) with permissive production code, then passed all
  22 after the minimal fixes. Safety probes use disposable sandbox roots only;
  tests never regenerate committed fixtures.
- Fresh disposition verifier (code-reviewer plus General/Python/Security guidance)
  verified the model and symlink fixes, but reproduced the related hardlink route:
  "The symlink route is fixed, but a hard link still bypasses the protected-fixture
  guarantee. With `--target invalid --force`, if an intended output such as
  `non-nfc.jsonl` is hard-linked to a file under the protected legacy/claims
  fixtures, `path.exists()` passes and `path.write_text()` truncates the shared
  inode (regen-corpus-fixtures.py:141–148)."
- Cycle two: `uv run pytest -q tests/unit/corpus_io_test.py -k hardlink` failed
  both invalid/schema cases (25 deselected), proving protected bytes changed.
  Swapping the direct writes for T-103 atomic_replace made both pass: protected
  inode unchanged, output detached. No new atomic helper or alias implementation.
- Second fresh disposition verifier confirmed all original runtime and writer
  fixes, plus the hardlink disposition. It proposed F-002-schema: "The committed
  JSON Schema accepts language tags with duplicate variants or extension
  singletons that the corpus models reject." A separate fresh candidate validator
  REJECTED this as an unstated schema/runtime parity requirement: the contract
  requires generated, drift-tested schemas, not reproduction of every custom
  runtime validator. The observed boundary mismatch is real; runtime checks are
  strict. Rejection recorded, not silently omitted. No remaining validated finding
  or deferred implementation work. Total review/fix/recheck cycles: two.

### Final checks before lifecycle finalization

- `uv run ruff check`: All checks passed!
- `uv run mypy`: Success: no issues found in 125 source files.
- `uv run pytest -v`: 1194 passed, 1 skipped in 45.93s.
- `mise run check`: Ruff passed; 332 files already formatted; mypy 125 files;
  1194 passed, 1 skipped in 46.76s; commit-message, push-message, author,
  differential/full mutation-results and mutation-configuration self-tests passed.
- All valid fixtures round-trip to identical bytes; literal header/passage wire
  assertions, 256-byte boundary, busy/alias wiring and protected no-clobber pass.
  Generator refuses existing output without --force. Legacy golden/spec diff empty.
- Fresh pre-finalization fetch still has accepted origin/main; spec hash unchanged;
  Taskrail validate passed with exactly this task active, T-038 still blocked.
  Hooks installed with `mise exec -- lefthook install`; maintainer identity checked.
- 2026-10-15T22:27:53Z: verification pass
- 2026-10-15T22:27:53Z: Reviewed and validated in two workflow-v3 cycles; final ruff, mypy, 1194 pytest (one skip), and mise check passed. No commands or legacy changes.
