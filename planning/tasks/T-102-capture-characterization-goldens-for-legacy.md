---
id: T-102-capture-characterization-goldens-for-legacy
title: Capture characterization goldens for legacy Vulgate paths and v1 claim exports
status: todo
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies: []
updated_at: "2026-10-14T05:42:59Z"
---

# T-102-capture-characterization-goldens-for-legacy Capture characterization goldens for legacy Vulgate paths and v1 claim exports

## Description

Commit characterization goldens for every legacy path the v0.3.0 refactors touch, captured
from current `main` before any USFX or claim code changes. Implements the first Delivery Order
step and the byte-identity promise of `specs/v0.3.0.md#usfx-normalization` ("keep their
options and byte-identical stdout and files") plus the v1 claim-sidecar goldens that
T-115 must keep loading. Today the USFX path is `cli.parse_usfx_to_df` with callers in
`preview_impl`, `update_csv_with_cloze`, `_update_apkg_with_cloze`, `load_corpora` (cloze) and
`validate_impl`; no golden CSVs or committed `*.claims.json` exist under `tests/fixtures/`.
This task changes no source under `src/`; moving the parser is T-108, rewiring is T-112,
claim sidecar v2 is T-115.

## Acceptance

- `tests/fixtures/goldens/legacy-vulgate/cases.json` lists each case (name, argv with
  `{FIX}`/`{TMP}` placeholders, expected exit code, golden stdout file, golden output
  files). `tests/unit/legacy_goldens_test.py` runs every case as a subprocess
  `[sys.executable, "-m", "latinitas_cards", ...]` with cwd = a tmp dir and env
  `COLUMNS=120 LINES=40 TERM=dumb NO_COLOR=1 TZ=UTC LC_ALL=C.UTF-8 PYTHONHASHSEED=0
  PYTHONIOENCODING=utf-8` (plus `PATH`, `HOME`=tmp, and `MUTANT_UNDER_TEST` passed through
  when set), ignores stderr, replaces the tmp dir path (raw and resolved) with `<TMP>` in
  stdout, and compares stdout and every listed output file byte for byte. Each case runs
  in a fixed-length directory under `/tmp` (`tempfile.mkdtemp(prefix="lg", dir="/tmp")`),
  and the harness rejects a case whose stdout contains the tmp path in a line longer than
  `COLUMNS` (Rich wrapping is not portable). At least these cases on a synthetic fixture:
  `preview --input input.csv --usfx sample.usfx.xml --limit 5` (and with
  `--multi-cloze-per-verse --max-examples 3`); `generate --input input.csv --usfx
  sample.usfx.xml --output out.csv` (default, `--append` over `existing.csv`, and with
  `--stopwords`, `--word-forms`, `--lemmas`, `--ignore-pattern`); `cloze --input forms.csv
  --output out.csv --corpus X` for `sample.usfx.xml` (auto), `sample.txt` (auto),
  `parallel.csv --latin-column la --parallel-mode include`, `--translation-lang en`,
  `--parallel-mode prompt` (non-TTY: translations ignored), `--difficulty easy`,
  `--include-column`/ `--exclude-column`, and two `--corpus` files mixed; `validate
  --input input.csv --usfx sample.usfx.xml` (exit 0) plus failing cases (no verses,
  malformed XML, missing `Front`; exit 1).
- `sample.usfx.xml` pins the lenient legacy behaviour: books out of ID order (book-ID sort),
  container and empty-milestone `book`, `c@id` and `c@n`, milestone `<v id/>…<ve/>` and the
  existing-test container form `<v n='1'>…</v><ve/>`, `<h>`, an `<f>` note and a `<wj>` span
  (text inline, fragments joined by a space), a duplicate verse (kept), an empty verse
  (dropped) and verse `3a` (integer 3). No DOCTYPE fixture and no `.jsonl` corpus under
  `auto` are captured (both change intentionally in T-112).
- `generate --usfx` with an `.apkg` input built at test time by
  `cli_test._create_colpkg_with_anki2` compares the decoded `notes.flds` values (JSON
  golden), not archive bytes.
- Full bundled Vulgate: `generate --input vulgate-words.csv --usfx data/lat-clementine.usfx.xml
  --output out.csv --max-examples 2` (fixed 40-word input) is pinned as
  `vulgate-generate.digest.json` (`sha256`, byte length, row count), not a committed CSV; a
  `preview --limit 5` stdout golden uses the same input. The 40-word CSV must pass the legacy
  `csv.Sniffer` header check (a 2-column one-word-per-row file is rejected with "CSV seems to
  have no header row"): use `Front,Back,Notes` with multi-word `Back` values. The two
  full-Vulgate cases are skipped when `MUTANT_UNDER_TEST` is set. `data/` is added to
  `[tool.mutmut].also_copy`, `tests/unit/mutation_config_test.py` and
  `scripts/check-mutation-config-test.sh` assert it.
- `tests/fixtures/goldens/claims-v1/{principal-parts,form-parsing,phrases}/` hold
  committed inputs, decision batches, the resulting version-1 `*.claims.json` sidecars and
  CSV exports. `tests/unit/claim_goldens_test.py` (a) replays the batches through `claims
  record --batch` on a tmp copy and asserts sidecar bytes equal the committed ones, and
  (b) runs the exports on a tmp copy with the committed sidecars and asserts CSV bytes
  equal the goldens: `generate --input source.csv --profile profile.json --output X
  --approve-fresh-import` once per recipe (`--recipe principal_part_completion`, `--recipe
  principal_part_recognition`), `form-parsing export cases.json X --approve-fresh-import`,
  and `phrases export deck.csv selected.json recipes.json X` (phrase-input and recipe
  sidecars).
- `tests/fixtures/goldens/CAPTURE` holds the capture commit SHA and a SHA-256 manifest of
  every file under `tests/fixtures/goldens/legacy-vulgate/` and `claims-v1/`; a test asserts
  each committed golden still matches its manifest digest and that no unlisted file exists in
  those two directories, so a regenerated golden fails CI unless the manifest is edited in the
  same, explained commit.
- Negative controls: altering one byte of a committed golden makes its test fail (shown in
  the TDD evidence, not committed); tests never write into `tests/fixtures/`.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the commit SHA of `main` the goldens were captured from and the Vulgate digest value.

## Implementation Notes

- Capture from unmodified `main`: `scripts/regen-legacy-goldens.py` (deliberate updates
  only, never called by tests) runs the same harness and writes the golden files and
  `CAPTURE`; commit its output in the same commit as the tests. Tests only read goldens.
  It is the only writer of `tests/fixtures/goldens/legacy-vulgate/` and `claims-v1/`, and
  it refuses to run unless `--i-know-this-rewrites-characterization-goldens` is passed and
  `git rev-parse HEAD` equals the capture commit recorded in `tests/fixtures/goldens/CAPTURE`
  (on the first capture, when `CAPTURE` does not exist yet, it records HEAD). Corpus
  fixtures use `scripts/regen-corpus-fixtures.py` (T-104), never this script.
- Principal-part inputs: a 3-verb `source.csv` (`ID,Lemma,Forms,Meaning`, e.g. amō, loquor,
  sum), a `profile.json` built like `tests/unit/review_commands_test.py` (`source_id_field`,
  both recipes) and a committed `source.csv.latinitas.json` with a fixed `source_scope`
  (as in `docs/examples/deck.csv.latinitas.json`) so no random `allocate_source_scope()` runs.
  Form-parsing inputs derive from `docs/examples/form-parsing-cases.json` with decisions
  removed (see `parsing_source`); phrase inputs follow `phrase_inputs_test.fixture`/
  `saved_example` with `docs/examples/phrase-inputs.json` and a committed recipe request.
- Every file a capture command writes is compared unless it embeds a random or clock value;
  each exclusion is listed in `cases.json` with its reason. If an export cannot be made
  deterministic from committed inputs, block this task with a follow-up instead of masking.
- Rich reads `COLUMNS` at console construction, so subprocesses (not CliRunner) are required;
  keep the subprocess case count near 20 to bound suite time.
- Size guard: fixtures, the `CAPTURE` manifest, two test modules (plus the manifest
  test) and one script; no `src/` change. File a follow-up
  rather than widen.
