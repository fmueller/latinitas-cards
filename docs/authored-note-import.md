# Authored note import format (schema version 1)

This corpus-independent input format describes study material already written by
a learner or agent. It does not parse Markdown or check Latin analyses. The
Python loader, identity reconciliation, and CLI validation, preview, and CSV
export are available without corpus resources or principal-part profile mappings.

Save UTF-8 JSONL: one JSON object on each physical line, with no header, comments,
or blank lines. A final newline is optional. An empty file is a valid empty input.
Every object requires integer `schema_version: 1` (not a string or boolean),
`kind`, a stable author-chosen `key`, nested `provenance`, `status`, and
`language_tag`. Unknown fields, versions, and duplicate object keys (including
inside provenance) are errors. Required text fields
must be strings containing non-whitespace text; content is preserved verbatim.

`provenance` requires `document` and `section` labels and optionally a `reference`
string or null. Labels and references are opaque: spelling, punctuation, and
whitespace are preserved, without corpus resolution or normalization.
`status` is exactly `include` or `skip`. `language_tag` describes learner-facing
content, for example `de` or `de-DE` (2–8 ASCII letters followed by optional
hyphen-separated 1–8 ASCII alphanumeric subtags).
Optional `tags` defaults to an empty array; supply an array of nonempty strings
without whitespace or control characters. Tag order and duplicates are preserved
at this boundary; later reconciliation handles deduplication.

## Kinds and examples

`vocab` requires `lemma` and `meaning`; optional `dictionary_form` is a string or
null (for example genitive/gender or principal parts).

```jsonl
{"schema_version":1,"kind":"vocab","key":"lecture:gratia","lemma":"grātia","dictionary_form":"grātiae f.","meaning":"Gnade; Dank","provenance":{"document":"Vorlesung","section":"Wortschatz","reference":"Lk 1,28"},"status":"include","tags":["lecture::1"],"language_tag":"de"}
```

`form` requires `text_form`, `base_form`, `analysis`, and `translation`; optional
`context` is a string or null. Use distinct keys for contextually distinct
occurrences, even if the text form is identical.

```jsonl
{"schema_version":1,"kind":"form","key":"lecture:abutere:1","text_form":"abūtēre","base_form":"abūtor","analysis":"2. Person Singular, Futur, Indikativ, Deponens","translation":"du wirst missbrauchen","context":"Quō ūsque tandem abūtēre?","provenance":{"document":"Vorlesung","section":"Formen","reference":"Cic. Cat. 1,1"},"status":"skip","language_tag":"de"}
```

`qa` requires `question` and `answer`; its optional citation is also stored in
`provenance.reference`.

```jsonl
{"schema_version":1,"kind":"qa","key":"lesson3:ablative","question":"Was ist ein Ablativ?","answer":"Ein lateinischer Kasus.","provenance":{"document":"Kursnotizen","section":"Fragen","reference":"Lektion 3"},"status":"include","language_tag":"de-DE"}
```

## Whole-file validation boundary

```python
from pathlib import Path
from latinitas_cards.authored_import import load_authored_import

result = load_authored_import(Path("notes.jsonl"))
for error in result.errors:
    print(error)  # physical line number, field, failed assumption
items = result.require_valid()  # raises AuthoredImportError if ANY error remains
```

The loader validates every row, including `skip` rows, before any filtering.
Encoding, malformed JSON, and schema errors are retained together, and later
valid rows still appear in `diagnostic_items` for preview reporting. These partial
items are **not an exportable selection**. Selection/export callers must call
`require_valid()` before filtering or touching output files. Successful items
carry loader-assigned `line_number` metadata; that field is forbidden in input.
Read/open failures propagate as `OSError`. This loader writes no output files.

## Stable identity and duplicate reconciliation

```python
from latinitas_cards.authored_identity import reconcile_authored_import

resolved = reconcile_authored_import("my-course", result)
notes = resolved.require_valid()  # loader errors AND duplicate conflicts block selection
for note in notes:
    print(note.latinitas_id, note.item.key, note.lines)
print(resolved.merged_duplicates)  # physical line groups for compatible duplicates
```

Choose a stable, nonempty collection namespace. Namespaces are used verbatim,
not normalized. Keys are Unicode NFC-normalized, outer whitespace is stripped,
and each run of Unicode whitespace is replaced by one ASCII space (Python
`str.split()` whitespace semantics). Case and punctuation are preserved:
`" lesson:\t1 "` and `"lesson: 1"` are one key; `"Lesson: 1"` is different.
Renaming a normalized key, namespace, or kind creates a different identity.

Identity reuses the v0.1.0 `latinitas-v2-` note-family contract: the source identity
is compact, non-ASCII-escaped JSON `["authored", namespace, kind]`, the object key
is the normalized key, and no source scope is supplied. Content, language, tags,
status, provenance, and line numbers never enter the digest. Correcting them on
a later import keeps the ID; conflicting duplicates within one import still fail.

Reconciliation covers all rows, including `skip`, before filtering. Same-kind,
same-normalized-key duplicates must agree exactly on required content, language,
document, and section. Optional `dictionary_form`, `context`, and reference can
be completed from another row when absent (missing, null, or empty string).
Different nonempty values conflict; whitespace-only optional strings are nonempty
and preserved verbatim. Tags are unioned and sorted; `skip` dominates `include`.
Merged keys are normalized, empty optional values become null, and notes are
sorted by kind and normalized key independently of input order. Each merged note
retains all contributing physical lines and the smallest line as item metadata.
Conflict diagnostics name both contributing lines and each differing field.
`diagnostic_notes` are partial reporting data, not an exportable selection; call
the reconciliation result's `require_valid()` before selecting or exporting.

## Validate and preview without writing files

Supply an explicit, stable collection namespace; no profile, corpus resources,
or principal-part field mappings are required:

```bash
uv run latinitas-cards authored validate notes.jsonl --namespace my-course
uv run latinitas-cards authored preview notes.jsonl --namespace my-course
uv run latinitas-cards authored preview notes.jsonl --namespace my-course \
  --kind vocab --section 'Lesson 3' --reference 'Lk 1,28' --tag lesson::3
```

Repeat a filter for OR matching within that dimension; different dimensions are
combined with AND. Section/reference/tag values match exactly, including whitespace.
Filters run after whole-file validation and duplicate reconciliation, so they see
merged tags and completed references. `skip` notes are counted but never selected.

Both commands report counts for reconciled nonconflicting notes by kind, section,
reference (`None` means absent), and status, plus merged line groups and all errors.
Invalid rows and conflicting groups are excluded from those counts and reported
as diagnostics. Preview additionally shows effective filters, selection size, and
one representative rendered front/back per matching kind (HTML shown as text).
An empty valid selection succeeds and is reported explicitly. Any validation or
conflict error anywhere, even in skipped or filtered-out rows, exits nonzero; cards
shown then are diagnostic only, with no exportable selection.

Neither command creates output files or modifies the input. The internal typed
`AuthoredPreviewResult` is separate from terminal formatting; consumers must call
`require_valid_selection()` before using notes as an exportable selection. There
is no stable JSON preview API in this release.

## Export selected notes and import into Anki

Keep the namespace explicit and unchanged across edits. Pre-create an output
directory, then export the same combined selection you reviewed:

```bash
mkdir -p authored-csv
uv run latinitas-cards authored export notes.jsonl --namespace my-course \
  --output-dir authored-csv --deck 'Latin::Authored' \
  --kind vocab --kind form --section 'Lesson 3' --reference 'Lk 1,28' --tag lesson::3
```

This selects vocabulary OR form, AND the exact section, reference, and merged
tag; `skip` is always excluded. Omit filters to export every included kind.
The command reports effective selection and writes `vocab.csv`, `form.csv`, and
`qa.csv` only for kinds with selected notes. It validates the entire file,
including skipped/filtered rows and conflicting duplicates, before any writes.
Invalid input exits nonzero without changing existing output or creating new
files. An empty selection reports zero and writes nothing, even if the output
directory is absent. Files left from an earlier, different selection are not
deleted: import only the files reported by the current successful command.
The input is never overwritten; symlink destinations are rejected.

CSV uses UTF-8, LF line endings, fixed schema field order, HTML-safe content,
and the shared v0.1.0 recoverable export boundary. Repeating a run with the same
input, namespace, deck, filters, and generator version produces identical bytes.
Headers specify comma separation, HTML, dedicated note type, target deck,
column names, and the special Tags column. Managed provenance is preserved;
input tags are merged and sorted. `Profile` is the fixed marker `authored`, not
a corpus/profile mapping. **Personal Notes is absent from all CSV columns.**

### First import

1. Back up Anki. Create the three dedicated note types using the exact field
   order and single `Recognition` template in [authored-note-types.md](authored-note-types.md).
   Headers preset existing models; they do not install fields or templates.
2. Pre-create `Latin::Authored` (or your explicitly chosen target deck).
3. Import each reported CSV into its named note type with **Allow HTML in
   fields** enabled. Map each managed column to the matching regular field and
   `Tags` to Anki's special tags destination, not a regular field.
4. Keep `LatinitasID` first; **leave Personal Notes unmapped**. Choose **Update
   existing notes when first field matches**, match scope **note type**. Confirm
   one intended Recognition card per note and check a representative answer.

### Re-import after edits

Edit managed content in JSONL, retaining kind, normalized key, and namespace.
Validate and preview again, then repeat the same export command. Import the
new CSV into the same dedicated note type with first-field Update enabled and
the same managed mappings; Personal Notes remains unmapped because it is not
offered as a column. Edited fields update the existing note, rather than adding
a duplicate. Changing key, kind, or namespace intentionally creates a new identity.
Removing or skipping an item does not delete a note already imported into Anki.
Mapped tags may replace destination-only tags when managed content changes;
back up those tags or defer import if they must be preserved. There is no live
collection conflict detection, merge, or automatic Anki write.

### Native Anki verification

On 2026-10-04, Anki 26.09.3's native backend (PyPI `anki==26.9.3`) on Linux
was exercised in a disposable collection using actual CLI-generated CSV files.
For **each** of vocab, form, and QA, first import created one note and one
Recognition card. A fresh `get_csv_metadata` + `import_csv` re-import with
first-field Update and note-type match scope changed Meaning, Translation, or
Answer respectively while retaining note ID, LatinitasID, card ID, target deck,
and seeded nonempty Personal Notes. Native card question/answer rendering and
HTML escaping were checked. Total remained three notes and three cards.

Reproduce this optional native gate (no live collection is opened):

```bash
uv run --with anki==26.9.3 python scripts/check-authored-anki.py
```

This is a real native-backend import check, not a mocked importer or a claim
that the desktop dialog was clicked. Manual desktop mapping remains necessary;
the normal test environment intentionally does not depend on Anki.
