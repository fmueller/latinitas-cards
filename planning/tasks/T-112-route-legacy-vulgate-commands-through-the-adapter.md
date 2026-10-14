---
id: T-112-route-legacy-vulgate-commands-through-the-adapter
title: Route legacy Vulgate commands through the adapter and add cloze jsonl corpora
status: todo
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies:
    - T-111-verify-full-clementine-vulgate-normalization
updated_at: "2026-10-14T05:42:59Z"
---

# T-112-route-legacy-vulgate-commands-through-the-adapter Route legacy Vulgate commands through the adapter and add cloze jsonl corpora

## Description

Route the legacy Vulgate commands through the hardened adapter reader and add normalized
JSONL corpora to `cloze`, per `specs/v0.3.0.md#usfx-normalization` (third paragraph and the
"Goldens and existing USFX tests pass after rewiring ..." / "`cloze --corpus x.jsonl`
golden" Acceptance sentences). After T-108, `usfx_adapter.legacy_projection` still uses
ElementTree; this task reimplements it on the expat event reader from T-109/T-110 in a
lenient mode that reproduces the old DataFrame exactly, so rejecting a
DTD/entity declaration or size/depth-unsafe XML is the only intended legacy change.
`preview`/`generate`/`validate`
gain no corpus option and no deprecation notice. Corpus-first cards are v0.4.0.

## Acceptance

- `legacy_projection` uses the shared expat reader (no `xml.etree` import remains in
  `usfx_adapter.py` or for USFX in `cli.py`; AST/import test) and keeps: case-insensitive
  `book|b`, `c|chapter`, `v|verse`, `ve`; `id|code|n` attributes; note and heading text
  inline; text fragments joined with one space then whitespace-collapsed; duplicates kept;
  empty verses dropped; integer chapter/verse from the first digit run; book-ID sort;
  `text_norm` via `normalize_latin`; `ValueError` on zero verses.
- The differential test from T-108 (`tests/unit/legacy_usfx_reference.py` oracle)
  passes on every golden and inline USFX fixture plus new lenient-edge fixtures (uppercase
  tags, `<b code>`, `<chapter n>`, a default-namespace document, text after `</v>` before
  `<ve/>`); any intentional divergence other than XML safety rejection fails the task.
- Every T-102 golden case passes byte-for-byte with no golden file modified, including the
  full-Vulgate `generate --usfx` digest; the existing USFX tests pass. The spy test still
  shows each legacy path calls `legacy_projection` exactly once.
- A DOCTYPE or entity declaration given to `preview --usfx`, `generate --usfx`, `cloze`
  (USFX) or `validate --usfx` exits 3 with an `unsafe_xml` message on stderr and writes no
  output file; malformed XML keeps its current legacy behaviour (validate: FAIL row, exit 1).
- The same legacy paths enforce T-110's 64 MiB and depth-64 limits: over-size/depth-65
  XML exits 3 (`input_too_large`/`depth_exceeded`) without output; size/depth boundary
  controls pass. These expected adapter errors never become a legacy validate FAIL/exit 1.
- `cloze --corpus-format` accepts `jsonl`; `auto` maps `.jsonl` to `jsonl` (previously
  `txt`; `--corpus-format txt` restores it, tested). Each passage is one context with
  `ref = citation.display`, also when mixed with USFX/txt/CSV corpora in one run (the
  existing formats keep their `book chapter:verse` refs; see notes), and each
  `translations[]` entry becomes `translation_<language>` for `--translation-lang` and
  `--parallel-mode include|ignore|prompt`. A corpus failing the header gate exits non-zero
  with the reader's message and writes no output.
- New committed golden `tests/fixtures/goldens/cloze-jsonl/` (`corpus.jsonl` with
  `translations` for `en` and `de`, written by `scripts/regen-corpus-fixtures.py --target
  cloze-jsonl` (T-104) through `corpus_io` and passing `corpus validate`) covers `cloze
  --corpus corpus.jsonl` (auto detection), `--translation-lang en`, `--parallel-mode
  include`, and a mixed `--corpus sample.usfx.xml --corpus corpus.jsonl` run, through the
  T-102 subprocess harness. Its cases live in `cloze-jsonl/cases.json` (never in
  `legacy-vulgate/cases.json`); the harness parametrizes over every `cases.json`. The test
  also makes literal assertions on the output CSV: the `ref` of a matched passage equals its
  `citation.display`, `translation_en` holds the fixture's English text, and auto detection
  picked `jsonl` (a `.jsonl` corpus parsed as `txt` would yield `corpus 1:N` refs).
- `cloze --help` lists `jsonl`; `preview`, `generate` and `validate` options and help are
  unchanged (help snapshot tests). No deprecation warning is printed anywhere.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record that `git diff --stat tests/fixtures/goldens/legacy-vulgate` is empty.

## Implementation Notes

- `collect_cloze_matches` and `generate_clozes_for_word` prefer `book/chapter/verse`
  columns over `ref`, so a jsonl frame must not reuse those columns: add a dedicated
  `display_ref` column that is checked first when non-null, leaving every other format's
  output (and the goldens) unchanged.
- Put the jsonl-to-DataFrame loader in a focused module (e.g.
  `src/latinitas_cards/corpus_cloze.py`); `load_corpora` only gains the dispatch branch.
  Use `normalize_latin(text)` for `text_norm` (not the `norm` field) so matching equals other
  formats.
- Malformed-XML errors must keep ElementTree's message format (`<expat error string>:
  line L, column C`, raised as a `ValueError` subclass caught where `ET.ParseError` was),
  because the `validate` golden prints it. Default-namespace tags must be compared as
  ElementTree saw them (`{uri}tag`), which the differential fixture pins.
- The legacy exit 3 for unsafe, over-size or over-depth XML is a typed adapter error
  caught in the four call sites; do not add envelopes or JSON output to legacy commands
  (v0.8.0 scope).
- Size guard: `usfx_adapter.py`, one cloze loader module and call-site edits in `cli.py`.
  File a follow-up rather than widen.
