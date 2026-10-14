---
id: T-111-verify-full-clementine-vulgate-normalization
title: Verify full Clementine Vulgate normalization against an independent oracle
status: todo
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies:
    - T-110-preserve-supported-usfx-markup-and-notes-and
updated_at: "2026-10-14T05:42:59Z"
---

# T-111-verify-full-clementine-vulgate-normalization Verify full Clementine Vulgate normalization against an independent oracle

## Description

Add the single full-corpus test of `specs/v0.3.0.md#usfx-normalization` ("One module-scoped
test normalizes the full bundled file ...") so USFX normalization is checked against the real
Clementine Vulgate and an independent oracle, not only small fixtures. It exercises the
strict reader from T-109/T-110 on `data/lat-clementine.usfx.xml`
(4,652,378 bytes; elements `usfx`, `book`, `h`, `c`, `v`, `ve` only; milestone verses) and
pins its output digest. It also confirms the mutation sandbox copies `data/`. Legacy command
rewiring is T-112; no production behaviour is added unless the oracle exposes a
defect in the reader, which is then fixed here with a focused regression fixture.

## Acceptance

- `tests/unit/usfx_full_corpus_test.py` has one module-scoped fixture that normalizes the
  bundled file once (in-process call of the adapter's normalize function with
  `corpus_id="vulgata-clementina"` and `data/lat-clementine.descriptor.json`, writing to a
  `tmp_path_factory` file); all assertions reuse it and no other test normalizes the full file.
- It asserts source SHA-256
  `05bcb3f669d9b36deab6cf26950b8a24c7609b292e5af1abfa51717677701950` (before and after: source
  bytes never change), 73 `work` records, 1,334 distinct (work, chapter) citation pairs,
  35,809 `passage` records, zero duplicate `passage_id`s, and `order` = 1..35,809.
- Independent oracle: a regex scan over the raw source text (`re` on `<book id="...">`,
  `<c id="..."/>`, `<v id="..."/>`, tracking the current book and chapter; no import from
  `usfx_adapter`, `corpus_model` or `cli`) builds the expected `passage_id` set
  (`vulgata-clementina:BOOK.CH.V`) and the expected per-book verse counts; both equal the
  normalized output.
- Spot checks against literal expected strings: GEN 1:1 text and `passage_hash` equal the
  pinned vector from `tests/fixtures/corpus/schema/hash-vectors.json`, the first five
  passages equal `tests/fixtures/corpus/usfx/gen-1-1-5.expected.jsonl` record for record,
  and the last passage is `vulgata-clementina:REV.22.21` with text `Gratia Domini nostri
  Jesu Christi cum omnibus vobis. Amen.`
- The output file's `sha256` equals the committed digest in
  `tests/fixtures/corpus/usfx/vulgate.digest.json` (`{"sha256", "bytes", "passages"}`);
  `corpus validate` logic (`corpus_validation`) over the same output reports no findings.
- The fixture records its wall-clock duration via `record_property("normalize_seconds",
  ...)` against the 15 s CI budget, asserts it is under 60 s, and the module is skipped when
  `MUTANT_UNDER_TEST` is set (`pytest.mark.skipif`).
- Assert (do not add) that `[tool.mutmut].also_copy` lists `data/` (added by T-102, whose
  `tests/unit/mutation_config_test.py` and `scripts/check-mutation-config-test.sh` already
  assert it).
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the measured normalization duration and the committed output digest.

## Implementation Notes

- The digest file is produced once by `scripts/regen-corpus-fixtures.py --target
  vulgate-digest` (T-104) and committed; the test never writes it. A digest change
  requires a deliberate regen and a commit explaining why.
- Keep the regex oracle deliberately simple and separate from the adapter: it must not
  share code, constants or helpers with `usfx_adapter.py`.
- Do not add a second full-corpus test (legacy full-Vulgate digest from T-102 is the only
  other reader of the file, through the legacy path).
- If the oracle disagrees with the adapter, fix the adapter with a minimal synthetic
  regression fixture in `tests/fixtures/corpus/usfx/` and keep the oracle unchanged.
- Size guard: one test module, one digest file and the mutation-config assertions. File a
  follow-up rather than widen.
