# Authored note import format (schema version 1)

This corpus-independent input format describes study material already written by
a learner or agent. It does not parse Markdown or check Latin analyses. The
Python loader and identity reconciliation are available now; CLI preview,
selection, and CSV export are separate planned work.

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
