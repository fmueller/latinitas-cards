# Authored note import format (schema version 1)

This corpus-independent input format describes study material already written by
a learner or agent. It does not parse Markdown or check Latin analyses. The
Python loader is available now; CLI preview, selection, identity reconciliation,
and CSV export are separate planned work.

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
