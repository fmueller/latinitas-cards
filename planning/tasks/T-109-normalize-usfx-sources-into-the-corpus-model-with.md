---
id: T-109-normalize-usfx-sources-into-the-corpus-model-with
title: Normalize USFX sources into the corpus model with descriptor provenance
status: completed
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies:
    - T-107-normalize-plain-text-sources-with-manifest
    - T-108-extract-usfx-parsing-from-cli-py-into-an-adapter
updated_at: "2026-10-16T00:16:17Z"
---

# T-109-normalize-usfx-sources-into-the-corpus-model-with Normalize USFX sources into the corpus model with descriptor provenance

## Description

Enable `corpus normalize SOURCE --adapter usfx --corpus-id ID --output FILE [--language TAG]
[--descriptor FILE] [--json]` for the structural USFX subset, writing a `latinitas-corpus`
v1 file with descriptor provenance, per `specs/v0.3.0.md#usfx-normalization` and Fixed
Contracts **Identity and hashing** (USFX `native_ref`), **Corpus records** and **Files**. It
adds a streaming stdlib `xml.parsers.expat` reader to `usfx_adapter.py` (beside
`legacy_projection` from T-108, which stays untouched), commits
`data/lat-clementine.descriptor.json`, and records provenance in
`docs/license-compatibility-audit.md`. Notes, headings, `p`/`q`/`wj`/`w` markup, entity and
size/depth hardening and `--allow-unknown` are T-110; the full-corpus oracle is T-111;
legacy commands keep ElementTree until T-112.

## Acceptance

- Supported elements: `usfx` (root attributes, including `xmlns:*` and
  `xsi:noNamespaceSchemaLocation`, are ignored), `book@id` as container or empty milestone,
  `h` (-> `work.title`), `c@id|n`, `v@id|n` in milestone (`<v id="1"/>...<ve/>`) and
  container (`<v id="1">...</v>`, optional trailing `<ve/>`) form, and `ve`. `id` wins over
  `n`. `native_ref = <book>.<chapter>.<verse>` with each attribute string encoded by
  `corpus_model.encode_ref_component` from NFC UTF-8 once (`3a` stays `3a`; `1.2`
  becomes `1%2E2`, `%2E` becomes `%252E`, `é` becomes `%C3%A9`); encoded refs must pass
  T-104's uppercase-escape grammar without decode/re-encode. Pin these values in synthetic
  normalization/validation round trips, including distinct raw `.` and `%2E` IDs; citation
  scheme `usfx-bcv`, parts book/chapter/verse with raw strings, `display` `"GEN 1:3a"`.
- Passages keep document order (no book sort), `order` 1-based; works in first-appearance
  order. Passage text = verse character data joined without inserted separators, then
  `canonical_text` (NFC, whitespace runs to one space, trimmed); empty verses exit 3
  (`empty_passage`); a duplicate `native_ref` exits 3 (`duplicate_source_ref`); a verse outside
  a book/chapter, an unterminated milestone verse at the next `v`/`c`/`book` or at document
  end, or non-whitespace text outside a verse exits 3 (`invalid_structure`).
- Any element outside this subset exits 3 (`unknown_element`, one diagnostic listing every
  element name with its count); malformed XML exits 3 (`invalid_xml`); a DOCTYPE exits 3
  (`unsafe_xml`) via expat `StartDoctypeDeclHandler`. Nothing is written on any failure.
- `tests/fixtures/corpus/usfx/gen-1-1-5.usfx.xml` (the source's bytes of
  `data/lat-clementine.usfx.xml` from the start, with its XML declaration and root
  attributes, through GEN 1:5's `<ve/>`, followed by `</book></usfx>\n`) normalizes with
  `--corpus-id vulgata-clementina --descriptor data/lat-clementine.descriptor.json` to bytes
  equal to the committed `gen-1-1-5.expected.jsonl`, and the output passes `corpus validate`.
  Independently of the snapshot, the test asserts literal values written in the test
  source: the five `passage_id`s `vulgata-clementina:GEN.1.1` ... `GEN.1.5`, GEN 1:1 `text`
  and `passage_hash` equal to the T-104 hash vector, GEN 1:3 text `Dixitque Deus : Fiat lux.
  Et facta est lux.`, `citation.display` `GEN 1:1` and `GEN 1:3`, and `work.title`
  `Genesis`.
- Synthetic fixtures under `tests/fixtures/corpus/usfx/` cover milestone and container
  books and verses, `c@n` and `v@n`, a `3a` verse, a `.` inside an id, a duplicate verse
  (exit 3), non-NFC input (stored NFC; hash equals the NFC fixture's), and a non-whitespace
  text run outside a verse (exit 3). Source file bytes are unchanged after every run.
- `data/lat-clementine.descriptor.json` (`latinitas-corpus-descriptor` v1) has edition
  "Biblia Sacra Vulgata Clementina", publisher "eBible.org", url
  "https://ebible.org/find/details.php?id=latVUC", license "Public Domain"; a test asserts
  these four values and that the header copies them. Without `--descriptor` they are null.
  `docs/license-compatibility-audit.md` gains a `## Bundled corpus provenance` section
  (prose or a table whose rows do not start with "| `"), naming file, SHA-256, descriptor
  values and "maintainer confirmation pending" (the release gate confirms); a doc test
  asserts it and `release_candidate_test.py` still passes unchanged (its audit parser
  counts every line starting with "| `" as a dependency row).
- `--manifest` or any approval flag with `--adapter usfx` exits 2; the cross-environment
  subprocess determinism test from T-107 is parametrized over the USFX fixture.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Verified and completed through Taskrail at 2026-10-16T00:16:17Z, after all gates.
- Base: origin/main `e86480c75aecb537b5343cb817cafd97e22e9f50`. Spec SHA-256:
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
  Initial validate was clean; next selected this task, with no active owner.
- Strict TDD: initial public CLI tests failed 16 cases (unsupported USFX) then
  passed 21. An explicit empty-container regression failed with invalid_structure
  instead of empty_passage, then passed (25 total focused cases).
- Dedicated Task loaded code-simplifier; no safe meaningful simplification found,
  no edits made. Initial mandatory chain passed: ruff, mypy (138 source files),
  pytest (1370 passed in 84.58s).
- Separate parallel code-reviewer Tasks covered General, Security and Python.
  General loaded ECC code-reviewer and common rules; Security loaded ECC
  security-reviewer, security-review and common security rules; Python loaded
  ECC python-reviewer and python-patterns. General reported:
  "No concrete task-relevant findings." Security was required for untrusted XML;
  Python for expat/state-machine semantics. No framework, database or other domain
  lane was applicable. Two specialists stayed within the default review budget.
- Fresh candidate-validation Task loaded code-reviewer, independently reproduced
  and validated both candidates below. None were rejected or deduplicated.

### Validated findings and dispositions

- SEC-001 (medium), verbatim finding: "USFX XML with an unsupported encoding
  declaration escapes the `invalid_xml` diagnostic path and is reported as an
  internal failure (exit 70), rather than a malformed-XML error (exit 3)."
  Evidence: parser caught only expat.ExpatError; bogus encoding raised LookupError
  and UTF-32 raised ValueError. Fixed by translating these parser failures into
  invalid_xml. Four public CLI regressions cover both encodings with existing
  and absent output, preserving source and output bytes or absence.
- F-109-1 (medium), verbatim finding: "Reject a USFX document that contains no
  works; the adapter currently accepts an empty root and writes a corpus that
  violates the Fixed Contract definition of a corpus as containing one or more
  works." Evidence: `<usfx/>` returned success with only header/corpus records.
  Fixed by requiring at least one work before building records. Two CLI
  regressions preserve existing output or absence and require invalid_structure.
- Both fixes used strict TDD: the six new cases failed before production edits
  (6 failed, 25 deselected); after the minimal fixes, the combined USFX/text/exit
  matrix suite passed 101 tests. No deferrals or new follow-up tasks.
- Fresh disposition-verification Task loaded code-reviewer in the required mode
  with General guidance. Both findings VERIFIED; 12 targeted tests passed,
  42 deselected. Conclusion: "No concrete task-relevant findings."
  One complete review/fix/recheck cycle; no unresolved findings.

### Final evidence

- Exact mandatory chain: `uv run ruff check` passed; `uv run mypy` passed
  (138 source files); `uv run pytest -v` passed (1381 tests in 78.07s).
- `mise run check` exited 0: ruff/format/mypy passed; 1381 tests passed in
  79.17s; commit-message, push-message, author, differential/full mutation and
  mutation-configuration guard suites passed. Legacy characterization goldens
  and release_candidate_test.py passed unchanged.
- Real subprocess CLI normalize, validate and rerun all exited 0. Validation
  counted header=1, corpus=1, work=1, passage=5. Genesis bytes matched the
  committed snapshot, SHA-256
  `f9fdb165d6f070b302a30ecf874c7ad989336e058c7b7cf9cfefd2a2a8ef1cc6`.
  Regenerated deliberately with `regen-corpus-fixtures.py --target usfx`;
  tests pin source-prefix bytes and independent literal IDs/text/hash/citations.
- Cross-environment subprocess tests cover cwd, TZ, locale and hash seed for
  both adapters. Additional disposable probes crossed 64-KiB chunk boundaries
  in UTF-8 (65607 bytes) and UTF-16LE/BE (131216 bytes), each yielding exactly
  65534 expected text characters without added separators or source mutation.
- Source SHA-256 remains
  `05bcb3f669d9b36deab6cf26950b8a24c7609b292e5af1abfa51717677701950`;
  spec hash unchanged. Product edits respect the task size guard. Legacy
  projection is untouched. Markup/full hardening, full-corpus oracle and legacy
  routing remain T-110/T-111/T-112; existing T-127 was not duplicated or changed.
  Descriptor provenance still requires maintainer confirmation at the release gate.
- After completion, validate returned state valid; next selected T-110 with
  off_spec=false. No next task was started; T-038 remains blocked.

## Implementation Notes

- Use `xml.parsers.expat.ParserCreate()` directly (`ET.XMLParser` does not expose
  `.parser`), feed the file in chunks, and emit events from a small state machine; no XML
  dependency is added. Design the event reader so T-110 can add handlers and T-112 can
  drive a lenient projection from the same events.
- Header: `adapter: "usfx/1"`, `norm_rules: "latin-fold/1"`, `source` from raw bytes
  (basename only); write via `corpus_io.write_corpus` with the alias/lock rules.
- Fixtures and the expected JSONL are produced once by `scripts/regen-corpus-fixtures.py
  --target usfx` (T-104) and committed; tests never regenerate them.
- Register new codes in the T-103 registry and docs table. Add the USFX section to
  `docs/normalized-corpus.md` (T-107).
- Size guard: the expat reader and strict builder in `usfx_adapter.py` plus wiring in
  `commands/corpus.py`. File a follow-up rather than widen.
- 2026-10-16T00:16:17Z: verification pass
- 2026-10-16T00:16:17Z: Reviewed structural USFX normalization; both validated findings fixed and independently verified. Mandatory chain and mise check pass; no changes to spec or legacy projection.
