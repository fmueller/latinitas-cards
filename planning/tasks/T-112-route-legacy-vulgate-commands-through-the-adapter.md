---
id: T-112-route-legacy-vulgate-commands-through-the-adapter
title: Route legacy Vulgate commands through the adapter and add cloze jsonl corpora
status: completed
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies:
    - T-111-verify-full-clementine-vulgate-normalization
updated_at: "2026-10-16T08:45:04Z"
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

## Execution Evidence

- Base and origin/main verified at f8db7202191bbf5cd05dec88cb8896bf6f44153f;
  active spec SHA256 fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704.
  `mise exec -- taskrail validate` returned `state valid`; `next --json` selected
  this task with `off_spec: false` before `start`.
- Strict TDD: `uv run pytest tests/unit/usfx_adapter_test.py -q` first reported
  18 failed / 9 passed (unsafe XML was accepted, size errors used exit 1, and
  the old XML import/error type remained). Shared expat rewiring then passed
  27 tests. Added APKG safety coverage and corrected the boundary control to
  exactly depth 64; no native Anki acceptance was reopened.
- JSONL RED: `uv run pytest tests/unit/corpus_cloze_test.py -q --tb=short`
  reported 10 failed / 1 passed: auto parsed JSON as text, explicit jsonl was
  unsupported, translations and gate rejection were absent, and the prescribed
  generator target was absent. GREEN: combined adapter/JSONL tests passed 38.
- Maintainer-authorized size-guard exception: added only the named
  `regen-corpus-fixtures.py --target cloze-jsonl` fixture target, reusing its
  symlink/overwrite preflight and `write_corpus` atomic safety. No generic
  generator refactor or acceptance/spec change. New cases use the existing
  subprocess harness with a directory parameter; legacy golden bytes untouched.
- Goldens captured only into `cloze-jsonl`; normalized corpus validation exited
  0 with counts header=1, corpus=1, work=2, passage=3. Literal CSV subprocess
  assertions independently pin citation, English translations and auto detection.
  Disabling auto JSONL detection deliberately made that literal test fail;
  restored behavior is covered by the full suite. Three help snapshots were
  independently compared byte-for-byte against source extracted from the base.
- Scope exclusions remain: T-127 schema exporter already filed; T-128/v0.8
  bounded-memory hardening unstarted; S2 dense-markup overhead is not a current
  demonstrated contract violation or gate; T-038 remains blocked. No spooling,
  extra caps, release actions, or active-spec edits.
- Initial mandatory chain: Ruff passed, mypy passed (142 files), pytest passed
  (1466 tests, one existing Pydantic warning, 103.23 s). Mypy test-only scalar
  typing issues were corrected before restarting the complete chain.
- Dedicated Task loaded `code-simplifier`: no changes; retained event-fragment
  handling and mixed-reference logic as necessary compatibility complexity.
  Its focused 49-test run, Ruff and mypy passed.
- Separate parallel read-only Tasks loaded `code-reviewer`: General with ECC
  code-reviewer (no companion specified), Security with ECC security-reviewer,
  security-review and common security rules, Python with ECC python-reviewer,
  python-patterns and common code-review rules. General and Python each stated
  "No concrete task-relevant findings." Security was required for the XML/input
  boundary; Python for parser/pandas semantics. No framework, database or other
  domain lane was materially triggered; two specialists stayed within budget.
- A fresh Task loaded `code-reviewer` candidate-validation mode, independently
  validated SEC-1, rejected no candidates, and found no duplicates. Verbatim:

> FINDING SEC-1 — security
> Severity: medium
> Evidence: `src/latinitas_cards/corpus_cloze.py:23` loads `record.citation.display` from the JSONL corpus; `src/latinitas_cards/cli.py:258-259` passes that value through as the display reference, and `cli.py:222` interpolates references into HTML without escaping. `Citation.consistent_display` in `src/latinitas_cards/corpus_model.py:174-189` checks that display matches the parts, but does not restrict HTML. Reproduced output: `{{c1::amo}} <span style='color:#888'>(<img src=x onerror=alert(1)>)</span>`.
> Finding: Escape JSONL citation display references before interpolating them into the cloze HTML output.
> Failure/impact: A crafted but schema-valid JSONL corpus can inject arbitrary markup into generated cloze fields; consumers that render the field as HTML may interpret that markup rather than display it as a citation.
> Recommended direction: HTML-escape the reference at the HTML rendering sink, and add a test showing that markup in a JSONL citation is emitted as text.

- SEC-1 disposition: fixed. Strict TDD command `uv run pytest
  tests/unit/corpus_cloze_test.py -k citation_is_text -q --tb=short` first
  failed both helper and real CLI rendering cases. JSONL references now have
  escaped HTML representations only at rendering, while collector `ref` remains
  the literal citation and legacy-format rendering is untouched. Focused tests
  passed 51; full chain passed 1468 tests; `mise run check` passed all guards.
- Fresh disposition-verification Task loaded `code-reviewer` with Security/ECC
  and security-review/common rules: SEC-1 resolved, but returned SEC-2. A second
  fresh candidate-validation Task confirmed it is distinct from SEC-1 and
  reproduced a schema-valid passage with unescaped markup. Validated wording:

> FINDING SEC-2 — security
> Severity: medium
> Evidence: `src/latinitas_cards/corpus_cloze.py:21-22` copies `record.text` directly into the JSONL DataFrame. `CorpusPassage.text` in `src/latinitas_cards/corpus_model.py:227` is an unrestricted string, while `src/latinitas_cards/cli.py:248-255` inserts matched source text into cloze markup and `cli.py:2201` writes it into an HTML-bearing field without escaping it. I confirmed the example `<img src=x onerror=alert(1)> Deus` passes `read_corpus` schema validation and produces `<img src=x onerror=alert(1)> {{c1::Deus}}`.
> Finding: Escape JSONL passage text before emitting it in the generated cloze HTML.
> Failure/impact: A schema-valid JSONL passage containing markup, such as `<img src=x onerror=alert(1)> Deus`, can put active markup into a generated cloze field when “Deus” is matched. Consumers rendering that field as HTML may interpret the markup.
> Recommended direction: Escape JSONL passage text at the HTML rendering boundary while preserving cloze markers, and add a test asserting markup in passage text is emitted as text.

- SEC-2 disposition: fixed in the second and final review/fix cycle. RED:
  `uv run pytest tests/unit/corpus_cloze_test.py -k passage_is_text -q --tb=short`
  failed four helper/CLI and single/multiple-cloze cases on literal `<img>`
  markup and `&`. GREEN: 55 focused tests passed after JSONL-only escaping of
  the generated cloze representation; source text, normalization and matching
  remain raw, and cloze markers remain intact. No findings deferred, rejected,
  or duplicate; no follow-ups created.
- Real subprocess manual matrix: DTD, external entity, depth 65 and size 64 MiB
  plus one each rejected across preview/generate/cloze/validate with exit 3,
  empty stdout and unchanged existing destination bytes. Exactly 64 MiB at
  depth 64 passed all four commands, including integer `GEN 2:7` output refs.
  All 20 subprocess checks passed; `git diff --stat
  tests/fixtures/goldens/legacy-vulgate` remained empty.
- Final checks after SEC-2: exact Ruff → mypy → pytest chain passed, with
  `1472 passed, 1 warning in 96.93s`; `mise run check` passed Ruff, format,
  mypy, all 1472 tests (94.71 s) and every policy/mutation guard. The one warning
  is the existing deliberately invalid Pydantic source_extra serialization case.
- Second fresh disposition-verification Task loaded `code-reviewer` with
  Security/ECC, security-review and common security rules; independently checked
  changed sinks, raw-input preservation, markers, tests and final gate log.
  Verdict: "SEC-1 — RESOLVED", "SEC-2 — RESOLVED", and "No concrete
  task-relevant findings." Exactly two review/fix cycles; no unresolved findings.
- At 2026-10-16T08:44:27Z, origin/main still matched the accepted base and the
  spec hash was unchanged; Taskrail validated and selected this active task
  (`off_spec: false`) before final lifecycle writes. Hooks installed with
  `mise exec -- lefthook install`; maintainer identity confirmed, no trailers.
- 2026-10-16T08:44:50Z: verification pass
- 2026-10-16T08:45:04Z: Verified pass at 2026-10-16T08:44:50Z after full gates and fresh independent disposition verification; 1472 tests, 55 focused cases, 20 real subprocess checks; legacy goldens unchanged; SEC-1 and SEC-2 fixed with strict TDD.
