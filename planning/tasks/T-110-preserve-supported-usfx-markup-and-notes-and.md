---
id: T-110-preserve-supported-usfx-markup-and-notes-and
title: Preserve supported USFX markup and notes and reject unsafe or unknown XML
status: todo
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies:
    - T-109-normalize-usfx-sources-into-the-corpus-model-with
updated_at: "2026-10-14T05:42:59Z"
---

# T-110-preserve-supported-usfx-markup-and-notes-and Preserve supported USFX markup and notes and reject unsafe or unknown XML

## Description

Extend the strict USFX reader of T-109 with the remaining supported markup, notes
and headings, `--allow-unknown`, and full XML hardening, per
`specs/v0.3.0.md#usfx-normalization` (first paragraph and the synthetic-fixture Acceptance
sentence) and the passage fields `notes`, `headings`, `source_markup`, `source_extra` of
Fixed Contracts **Corpus records**. Today the strict reader accepts only `usfx`, `book`, `h`,
`c`, `v`, `ve` and rejects a DOCTYPE; the legacy ElementTree projection is unaffected here
(T-112 applies the hardening to it). The full-corpus oracle is T-111.

## Acceptance

- `p` and `q` are transparent containers (their text is passage text; no record);
  `wj` text stays in `text` and is recorded in `source_markup` as `{"element": "wj", "span":
  [start, end], "attributes": {}}` with code-point offsets into the canonical `text`; `<w>`
  text stays in `text` and its attributes (e.g. `s`, `l`, `x-morph`) are kept verbatim as
  strings in a `source_markup` entry with element `w` and span. Spans are checked by slicing.
- Notes `f`, `x`, `fe` and headings `s` leave `text`: each becomes one entry in passage
  `notes` (`{"kind": "f"|"x"|"fe", "text": ...}`, canonicalized, descendants flattened in
  document order and never checked as unknown) or `headings` (`{"text": ...}`); a note
  attaches to the verse it occurs in, a heading attaches to the next passage. A note outside
  any verse (for example between `<ve/>` and the next `<v/>`) attaches to the next passage,
  like a heading; with no next passage in the work it exits 3 `invalid_structure`. Nested
  `<f><fr>1:1</fr><ft>...</ft></f>` flattens without an `unknown_element` failure.
- An unknown element fails with exit 3 (`unknown_element`, one diagnostic listing names
  and counts, e.g. `zz=2, foo=1`) and writes nothing; with `--allow-unknown` the run exits 0,
  keeps the element's text in `text`, records `{"element", "attributes", "span"}` in the
  passage's `source_extra`, and reports the same names and counts as a `severity: "warning"`
  diagnostic (stderr line in text mode).
- Hardening, each exit 3 with nothing written: a DOCTYPE (`unsafe_xml`), an internal or
  external entity declaration (`unsafe_xml`, via `EntityDeclHandler`), an external entity
  reference (`unsafe_xml`, `SetParamEntityParsing(XML_PARAM_ENTITY_PARSING_NEVER)` and a
  rejecting `ExternalEntityRefHandler`), input over 64 MiB (`input_too_large`, checked from
  `stat` and while streaming, without reading the whole file), element depth over 64
  (`depth_exceeded`). The over-size test uses a sparse or generated tmp file, not a
  committed one; the depth fixture has depth 65 and a control with depth 64 passes.
- Synthetic fixtures in `tests/fixtures/corpus/usfx/markup/` cover milestone and container
  books and verses, `c@n`, `p`, `q`, `wj`, `w` attributes, nested `f`, `x`, `fe`, `s`, a `3a`
  verse, a duplicate verse (exit 3), an unknown element (failure, then `--allow-unknown`),
  non-NFC input, a DTD and an entity declaration; each has a committed expected JSONL or
  expected exit/code pair, and every success output passes `corpus validate`.
- The billion-laughs fixture (entity declarations) fails fast: the test asserts exit 3
  within a 5 s subprocess timeout and that no entity text appears in any output.
- The JSON Schema for passages (`docs/schemas/latinitas-corpus.v1.schema.json`) constrains
  `notes`, `headings`, `source_markup` and `source_extra` item shapes; the drift test passes.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- Extend the expat event reader in `src/latinitas_cards/usfx_adapter.py`; keep markup
  span bookkeeping in raw-offset space, then map spans through the whitespace
  canonicalization (build an offset map while collapsing whitespace) so spans index the
  final `text`.
- Hardening uses only stdlib expat handlers; add no dependency. Document the mechanism in
  `docs/license-compatibility-audit.md` or the USFX section of `docs/normalized-corpus.md`
  in one line.
- If the pinned `corpus_model` passage models need the item shapes above, change only
  those optional fields; `passage_hash` still covers only `text`.
- Expected JSONL files are snapshots written once by `scripts/regen-corpus-fixtures.py
  --target markup` (T-104); tests never regenerate them. In addition, each markup fixture
  test asserts at least one literal `source_markup` span (checked by slicing to a literal
  string), one literal note `{kind, text}` and one literal heading attachment `passage_id`,
  written in the test before the implementation.
- Decision: with `--allow-unknown` the unknown element's text stays in `text` so
  `passage_hash` covers it; `source_extra` records element, attributes and span. Unknown
  character-style wrappers would otherwise drop verse text; do not "fix" this.
- Size guard: changes in `usfx_adapter.py`, the passage model fields and fixtures. File a
  follow-up rather than widen.
