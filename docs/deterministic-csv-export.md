# Principal-part preview and deterministic CSV export

The confirmed profile workflow uses the existing `preview` and `generate` entry points
with `--profile` and is the v0.1.0 deck-first path. The legacy USFX corpus path is
experimental, requires `--usfx`, and cannot be combined with `--profile`.
The profile path never writes the source CSV/APKG/COLPKG or a profile. For an approved
ID-less CSV manifest, it persists the manifest at the source-side path described below.

```bash
uv run latinitas-cards preview \
  --input source.csv \
  --profile .latinitas/profile.json

uv run latinitas-cards generate \
  --input source.csv \
  --profile .latinitas/profile.json \
  --output generated-principal-parts.csv
```

Both commands use the same typed generation result. `preview` shows representative lemma,
principal-parts, and provenance values before any output is written, followed by source
entry, object, exported-note, card, zero-eligible-note, skipped, and ambiguous counts.
Structured skip and manifest-review reasons identify the source row and
failed assumption. `generate` renders that same preview first and writes only after the result
is safe to export. The structured result is internal in v0.1.0; it is not a stable JSON CLI
format.

One CSV row is one coherent learning object (one note with zero or more eligible cards), not
one exercise: do not confuse a row count with a card count. Both initial recipes render as
conditional sibling cards of that one shared note: each card has a frozen semantic key
(recipe plus confirmed role) mapped to a documented template slot with its own
`Enabled`/`Prompt`/`Answer` note fields, and the entire card front is guarded by the
per-card `Enabled` field so static labels never create blank cards. Profile recipe
selection changes card eligibility only — never note identity, slot order, or template
ordinals. Notes whose layout roles have no supported slot stay valid objects but yield zero
eligible cards; they are omitted from the CSV (native import would otherwise create a blank
note) and reported as a distinct zero-eligible count, separate from parser skips.

## CSV source-scope bootstrap

Every CSV source — with a stable source-ID column or with an ID-less manifest —
needs a committed unique source scope before stable identities exist. An
uninitialized `preview` reports that scope confirmation is required and mints
nothing. The first approved export allocates and commits the scope together
with the assignments and the output:

```bash
uv run latinitas-cards generate \
  --input source.csv \
  --profile .latinitas/profile.json \
  --output generated-principal-parts.csv \
  --approve-scope
```

Later exports reuse the persisted scope without the flag. A missing or legacy
unscoped sidecar is never silently replaced: `--approve-scope` is also the
explicit fresh-start or legacy-migration confirmation. `preview` rejects
`--approve-scope` because a read-only preview cannot commit identity state.

## Declared legacy note types

Pre-release collections may still carry notes from the retired per-exercise
model (see [legacy-transition.md](legacy-transition.md)). Exporting new-model
rows into one of those note types would silently reinterpret the legacy model,
so both `preview` and `generate` accept a repeatable `--legacy-note-type`
option naming legacy note types from your destination inventory:

```bash
uv run latinitas-cards generate \
  --input source.csv \
  --profile .latinitas/profile.json \
  --output generated-principal-parts.csv \
  --legacy-note-type "Latinitas Legacy Exercise" \
  --approve-scope
```

When the effective profile's generated note type matches a declared legacy
note type, the run is rejected outright before any output, identity state, or
checkpoint is written. Latinitas cannot detect your destination's note types
in v0.1.0; the guard covers exactly the note types you declare. A declaration
protects you only when it matches the generated note type exactly apart from
leading/trailing whitespace (the comparison is case-sensitive and interior
spacing matters), so copy the name from your destination inventory rather
than retyping it.

## ID-less CSV manifests

Profiles using the `manifest` source-identity strategy default to the sidecar
`source.csv.latinitas.json` beside the input source, not beside the generated output. The
sidecar also persists the unique source scope. A
valid manifest automatically reuses an unchanged unique row fingerprint. The first run
requires the scope approval above plus explicit allocation approvals, for example:

```bash
uv run latinitas-cards generate \
  --input source.csv \
  --profile .latinitas/profile.json \
  --output generated-principal-parts.csv \
  --approve-scope \
  --approve-allocation 0 \
  --approve-allocation 1
```

Approval row numbers are zero-based source-record indexes. For an edited, duplicate,
unmatched, or stale-manifest review, use `--approve-reuse ROW=SOURCE_ID` only when you have
confirmed that the row is the same source entry, for example:

```bash
uv run latinitas-cards generate \
  --input source.csv \
  --profile .latinitas/profile.json \
  --output generated-principal-parts.csv \
  --approve-reuse 2=existing-source-id
```

Use `--approve-allocation ROW` when the row should receive a new source identity instead.
Both options are repeatable. Unresolved reviews block export; no identity is silently
transferred or allocated. Keep the generated CSV and the source-side manifest together
when backing up or moving a project. The CSV, manifest, and card-evidence checkpoint are
staged together.
For caught process/I/O failures and catchable interruptions such as Ctrl-C, the exporter
makes a best-effort attempt to restore the prior state before the failure or cancellation
is re-raised. Rollback is not durable pair-atomicity: a crash or power loss between the
replacements can leave one file new and the others old. Backups are deleted only after a
confirmed commit or confirmed recovery. If rollback fails, the error or interruption message
reports the affected destinations and retained backup locations; if rollback is itself
interrupted, no message may be emitted, and the retained `.backup.*` files in the output and
state directories hold the recoverable prior bytes. Recover manually from those files
and do not blindly retry or delete them. Use `--manifest` to select another sidecar path.

## Prior-export card-evidence checkpoint

Scoped CSV sources and package (APKG/COLPKG) sources persist a versioned prior-export
checkpoint beside the source — `source.csv.latinitas-cards.json` or
`deck.apkg.latinitas-cards.json`. It is committed with the output and the identity
manifest through the same recovery boundary and records exported state only: the source
scope, note family/schema, and template-slot registry it is bound to, each exported
object's `LatinitasID` with its eligible card keys, and the fingerprint of the committed
CSV. It is evidence about what Latinitas exported, never proof that a destination
collection imported it or still contains those cards; the destination-aware baseline is
v0.2.0 work. Package sources commit the checkpoint without an identity manifest; their
globally scoped GUID identities keep one shared binding.

Because a package source keeps no local record that could prove a first export, a missing
checkpoint next to one is never read as an empty prior card set: the first package export,
like every recovery from missing, corrupt, or incompatible state, requires the same
explicit fresh-import confirmation below. Renaming or re-downloading a package source does
not carry the sidecar along, so the confirmation is asked for again rather than silently
exporting against assumed-empty evidence.

Regeneration compares current eligibility against this retained evidence. When a previously
exported card key is no longer eligible — through data loss, recipe deselection, or a
shared-field change that removes a card's required data — the whole affected note row is
withheld from the output and reported under `Card eligibility reviews` instead of clearing
fronts, deleting cards, or resetting schedules. Wording and gloss updates that keep every
exported key eligible still export normally, and adding a supported recipe enriches the
same note while surviving keys keep their slots. Objects absent from the current export
(removed rows, parser failures) keep their last safe evidence; a partial export never
asserts their cards disappeared. A read-only `preview` never advances the checkpoint.

A missing, corrupt, or incompatible checkpoint is never read as an empty prior card set:
the export stays review-only until you either recover/review the state or explicitly
confirm a fresh import:

```bash
uv run latinitas-cards generate \
  --input source.csv \
  --profile .latinitas/profile.json \
  --output generated-principal-parts.csv \
  --approve-fresh-import
```

`preview` rejects `--approve-fresh-import` for the same reason it rejects
`--approve-scope`: a read-only preview cannot commit or replace retained state.

## Source tag inheritance

For APKG/COLPKG sources, every generated learning-object note inherits
all valid tags of its parent Anki note. Inherited tags are read from the source
note's own tag metadata only; they are never inferred from neighboring notes or
unioned across the source deck. Configured tags (profile defaults, saved profile
values, or `--tag` CLI overrides) are additive: they are appended after the
inherited tags and never replace them. The combined list removes exact duplicates
deterministically, keeping first occurrence, and preserves source order,
hierarchy separators (`::`), case, and Unicode names. An untagged parent keeps
the existing configured-tags-only behavior.

Tags are metadata, not identity: changing tag membership or order never changes
`LatinitasID` values or source identities, and repeated generation stays
byte-deterministic. The preview `Tags:` line and the exported `Tags` column are
produced from the same combined list.

A source tag that contains whitespace or control characters cannot be inherited
safely. Such a parent is skipped with a structured `invalid_source_tags` skip
that names the source identity, the note location, and the offending tag
position; nothing is dropped silently. Fix the tag in the source deck and rerun.
The same skip applies to inherited tags containing `&`, `<`, or `>`: the export
enables Anki's Allow-HTML import mode, and a tag carrying live markup would be
stored verbatim and rendered unescaped by Anki's `{{Tags}}` template filter.
These characters are rejected with an actionable diagnostic rather than escaped,
because Anki stores tag names verbatim and escaping would silently rename the
tag. Locally configured tags are authored input and keep their existing
semantics; only tags inherited from an untrusted source package are subject to
the stricter check.

Supported source-tag boundary: only native APKG/COLPKG note tags are inherited.
CSV rows keep configured tags only — no CSV column is guessed to be Anki tag
metadata, and CSV profile mappings are unchanged.

Native tag-import verification (v0.1.0, Anki 26.09.3, historical evidence for the
retired per-exercise note model): combined inherited + configured tags were
verified against a native Anki 26.09.3 collection by importing the generated CSV
through Anki's own import backend
(`get_csv_metadata` + `import_csv` with `#html` and `#tags column` honored),
the same calls the import dialog makes; the dialog's UI was not click-driven
during this run. On the first import, all 24 generated per-exercise notes of the
old model were created with
exactly the combined tag sets from the `Tags` column, `LatinitasID` values
matched the CSV, and no duplicates existed. Anki stores note tags in its own
canonical order (padding the stored tag string with spaces), so stored tag
order differs from the CSV column order while the tag sets are identical. On a
repeat import with update-when-first-field-matches and match scope note type,
no duplicate notes were created and every `LatinitasID` stayed stable, but the
imported `Tags` column replaced each note's tag set instead of merging it: a
tag manually added to a note before the repeat import was removed by the
import, and a combined tag manually removed from a note was restored from the
CSV. That backend-path observation did not isolate a tag-only change, so it
does not by itself establish what happens when every managed field is
unchanged.

Native tag-boundary verification (v0.1.0, Anki 26.09.3, current learning-object
model): the release gate re-probed this boundary through Anki's native import
dialog on a disposable synthetic collection of the two complete objects
(dīcere and ferre; Debian Linux). A tag-only reimport whose CSV row was
unchanged in every managed field was reported by Anki as Skipped and left the
manually added destination-only tag in place — notes, cards, and review logs
stayed identical. A follow-up reimport whose row also changed a managed gloss
updated that note and removed the manually added tag, while every card and
review-log row stayed identical and the other note's Personal Notes were
preserved. The manual-tag removal on a managed-field change is reproduced; the
tag-only Skipped outcome is an observed Anki 26.09.3 dialog boundary, not a
LatinitasCards guarantee. Users must still treat every reimport as
tag-destructive, because any row that changes a managed field re-asserts the
exported tag set.

## Anki text import

The output is UTF-8 CSV with deterministic `\n` line endings. It uses Anki text-file headers
for `#separator`, `#html`, `#notetype`, `#deck`, `#tags column`, and `#columns`. The first
regular column is always `LatinitasID`; it is derived from the immutable source identity, the
persisted CSV source scope where applicable, and the reviewed learning-object key — not from
recipe selection, card roles, prompt text, glosses, HTML, tags, or local Anki IDs. The export
columns are derived from the authoritative note schema:
`LatinitasID, Lemma, Principal Parts, Meaning, Tags, Source ID, Source Scope, Source Kind,
Source Location, Source Path, Note Schema, Generator, Profile`, followed by one
`Enabled`/`Prompt`/`Answer` triple per frozen template slot of the two initial recipes.
The configured and inherited
tags are emitted in the `Tags` column (column 5) as Anki's special tags metadata,
not as a regular note field; provenance fields are included for auditing. `Source Path` is
intentionally blank to avoid leaking absolute paths. The profile's language tag is explicit
(`de` for the approved German content); it does not localize CLI controls or diagnostics.
The user-owned `Personal Notes` field is deliberately **not** a CSV column: generated import
data covers only managed fields, so a repeat import can never offer an empty personal value
for accidental overwrite. Generated CSVs and manifests still contain source-derived text,
stable IDs, and provenance; treat them as sensitive and redact them before sharing.

Before the first import, create the dedicated note type named by the profile (normally
`Latinitas Principal Parts`) in Anki. Create regular fields for every `#columns` name except
`Tags`, which is the special tags column, plus a trailing user-owned `Personal Notes` field.
The exact versioned field list, the copyable guarded front/back templates and CSS for all ten
card slots, synthetic missing-form examples, and the safe first/repeat import checklist are
published in [reference-note-type.md](reference-note-type.md); that reference setup is checked
against this export contract by tests. CSV import headers can preset an existing note
type and deck, but they do **not** create note types, fields, or templates. The `#deck`
header selects or presets the target. The current Anki manual documents that header as
presetting an existing deck and documents missing-deck creation for a deck column; a missing
target may still be created in some import flows, but do not rely on that when hierarchy or
settings matter—pre-create it. The header is not a template/schema definition. Enable
**Allow HTML in fields**, map `Tags` to Anki's tags column rather than to a note field, and
map every other regular column to the corresponding field. `Personal Notes` has no CSV
column, so the import dialog offers nothing to map to it.

For repeat imports:

1. Select the same dedicated note type.
2. Keep `LatinitasID` as the first/matching field and choose **Update existing notes when
   first field matches**. Use match scope **note type** (or **note type and deck** when that
   is an intentional local policy).
3. Map the managed columns (`Lemma`, `Principal Parts`, `Meaning`, `Tags`,
   provenance, and generation metadata) as before. `Personal Notes` stays unmapped because
   the generated CSV never contains it; no remembered Ignore selection is required.
4. Keep **Allow HTML in fields** enabled. Anki's manual documents that matching notes are
   updated in place, remain in their current decks, and preserve scheduling when updating is
   enabled. This behavior was verified in a disposable collection with native Anki Desktop
   26.09.3: repeat imports through freshly opened dialogs updated changed managed content,
   preserved personal notes, review history and scheduling, kept deck placement and stable
   `LatinitasID` values, and created no duplicate notes.

Older CSVs generated before this contract may still contain a `Personal Notes` column. When
importing such a file, map that column to `(Nothing)` / **Ignore field** on **every** repeat
import; Anki's previous selection may not persist between imports, and a mapped empty value
would overwrite personal notes.

Anki's authoritative text-import behavior is documented in the
[Anki Manual: Text Files](https://docs.ankiweb.net/importing/text-files.html), including
UTF-8, HTML, first-field duplicate matching, update behavior, and supported file headers.
The generated CSV is the auditable file boundary; it does not write a live Anki collection.
