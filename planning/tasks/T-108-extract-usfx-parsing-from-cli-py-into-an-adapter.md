---
id: T-108-extract-usfx-parsing-from-cli-py-into-an-adapter
title: Extract USFX parsing from cli.py into an adapter module with a legacy projection
status: completed
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies:
    - T-102-capture-characterization-goldens-for-legacy
updated_at: "2026-10-15T23:47:08Z"
---

# T-108-extract-usfx-parsing-from-cli-py-into-an-adapter Extract USFX parsing from cli.py into an adapter module with a legacy projection

## Description

Move the legacy USFX parser out of `cli.py` into a new adapter module as
`usfx_adapter.legacy_projection`, a pure red/green refactor required by
`specs/v0.3.0.md#usfx-normalization` ("After the goldens, the USFX code moves from `cli.py`
into the adapter module (red/green TDD). Legacy commands call its `legacy_projection`") and
the AGENTS.md refactoring direction. Today `cli.parse_usfx_to_df` (ElementTree, lenient tags
and attributes) is called from `_update_apkg_with_cloze`, `update_csv_with_cloze`,
`preview_impl`, `load_corpora` and `validate_impl`. Behaviour must not change at all in this
task: the T-102 characterization suite is the oracle. XML hardening, the expat reader and
DTD rejection arrive with T-109/T-110 and are applied to the legacy path by T-112; `cloze
--corpus-format jsonl` is also T-112.

## Acceptance

- RED first: a test asserts `latinitas_cards.usfx_adapter.legacy_projection` exists and that
  `cli.py` defines no `parse_usfx_to_df` (AST scan of `src/latinitas_cards/cli.py` for a
  `FunctionDef` of that name); it fails before the move and passes after.
- `legacy_projection(path: Path) -> pd.DataFrame` returns exactly what `parse_usfx_to_df`
  returned (columns `book, chapter, verse, text, text_norm`, book-ID sort, integer refs,
  duplicates kept, empty verses dropped, `ValueError` on zero verses, `ET.ParseError` on
  malformed XML). A frozen verbatim copy of the pre-move function lives in
  `tests/unit/legacy_usfx_reference.py` (test-only oracle, never imported by `src/`), and
  a differential test asserts `DataFrame.equals` on every USFX file under
  `tests/fixtures/goldens/legacy-vulgate/` and on the inline fixtures of
  `tests/unit/cli_test.py`.
- Spy test: monkeypatching `usfx_adapter.legacy_projection` with a recording wrapper shows it
  is called exactly once by each of `preview --usfx`, `generate --usfx` (CSV input),
  `generate --usfx` (`.apkg` input), `cloze --corpus x.usfx.xml`, `cloze --corpus-format usfx`
  and `validate --usfx`; `cli.py` resolves the function through the module attribute
  (`usfx_adapter.legacy_projection(...)`), not a `from ... import` binding, so the spy works.
- Every T-102 golden case (`tests/unit/legacy_goldens_test.py`, including the full-Vulgate
  digest) and every existing USFX test in `tests/unit/cli_test.py` and
  `tests/unit/principal_part_command_test.py` pass unchanged; no golden file is modified
  (`git diff --stat tests/fixtures/goldens` is empty in the task evidence).
- No other behaviour changes: option names, help text, stdout and exit codes of the four
  legacy commands are untouched (help-text test in `cli_test.py` still passes).
- `src/latinitas_cards/usfx_adapter.py` imports nothing from `cli.py`;
  `normalize_latin` moves with it (or into a small shared module) and `cli.py` re-exports
  the name so existing imports keep working.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Cite the RED run (failing structure test) and the GREEN run timestamps.

## Implementation Notes

- Move code, do not rewrite it: copy the function body verbatim, rename, keep ElementTree
  for now. Any cleanup beyond the move waits for T-112.
- `normalize_latin` is used by cloze matching across `cli.py`; avoid a circular import by
  placing it in `usfx_adapter.py` (or `latin_text.py`) and importing it into `cli.py`.
- Keep `build_bucket_index`, cloze helpers, `load_corpora` and the `*_impl` callbacks in
  `cli.py`; only the parser moves. If `load_corpora` must change, change only the call.
- Do not edit `tests/fixtures/goldens/`. If a golden fails, the refactor is wrong.
- T-112 later changes the malformed-XML exception type, so do not pin `ET.ParseError` in new
  tests beyond this task's differential test.
- Size guard: one new module and call-site edits in `cli.py`. File a follow-up rather than
  widen.
- 2026-10-15T23:47:08Z: verification pass
- 2026-10-15T23:47:08Z: Behavior-preserving parser/normalizer extraction with frozen verbatim oracle, differential fixtures and six command spies. RED 2026-10-15T23:34:28Z; GREEN 2026-10-15T23:35:50Z. Focused 91 and full 1344 tests passed; ruff/mypy and mise run check passed. Dedicated simplifier, separate General/Python/Security review lanes, fresh candidate validation and fresh disposition verification: no concrete task-relevant findings; one cycle, no deferrals. All T102 goldens and full Vulgate digest unchanged; no fixture edits. Verification run recorded immediately before completion.
