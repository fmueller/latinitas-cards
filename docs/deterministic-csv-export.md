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

Both commands use the same typed generation result. `preview` shows representative prompt,
answer, and provenance values before any output is written, followed by generated, skipped,
and ambiguous counts. Structured skip and manifest-review reasons identify the source row and
failed assumption. `generate` renders that same preview first and writes only after the result
is safe to export. The structured result is internal in v0.1.0; it is not a stable JSON CLI
format.

## ID-less CSV manifests

Profiles using the `manifest` source-identity strategy default to the sidecar
`source.csv.latinitas.json` beside the input source, not beside the generated output. A
valid manifest automatically reuses an unchanged unique row fingerprint. The first run
requires explicit allocation approvals, for example:

```bash
uv run latinitas-cards generate \
  --input source.csv \
  --profile .latinitas/profile.json \
  --output generated-principal-parts.csv \
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
when backing up or moving a project. The CSV and manifest are staged together.
For caught process/I/O failures, the exporter makes a best-effort attempt to restore the
prior pair. Rollback is not durable pair-atomicity: a crash or power loss between the two
replacements can leave one file new and the other old. If rollback fails, inspect reported
destinations and retained `.backup.*` files, recover manually, and do not blindly retry or
delete them. Use `--manifest` to select another sidecar path.

## Source tag inheritance

For APKG/COLPKG sources, every generated completion and recognition note inherits
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

Native tag-import verification (v0.1.0, Anki 26.09.3): combined
inherited + configured tags were verified against a native Anki 26.09.3
collection by importing the generated CSV through Anki's own import backend
(`get_csv_metadata` + `import_csv` with `#html` and `#tags column` honored),
the same calls the import dialog makes; the dialog's UI was not click-driven
during this run. On the first import, all 24 generated notes were created with
exactly the combined tag sets from the `Tags` column, `LatinitasID` values
matched the CSV, and no duplicates existed. Anki stores note tags in its own
canonical order (padding the stored tag string with spaces), so stored tag
order differs from the CSV column order while the tag sets are identical. On a
repeat import with update-when-first-field-matches and match scope note type,
no duplicate notes were created and every `LatinitasID` stayed stable, but the
imported `Tags` column replaces each note's tag set instead of merging it: a
tag manually added to a note before the repeat import was removed by the
import, and a combined tag manually removed from a note was restored from the
CSV. Repeat imports therefore re-assert the generated tag set; unrelated
manual tags on generated notes do not survive a repeat import.

## Anki text import

The output is UTF-8 CSV with deterministic `\n` line endings. It uses Anki text-file headers
for `#separator`, `#html`, `#notetype`, `#deck`, `#tags column`, and `#columns`. The first
regular column is always `LatinitasID`; it is derived from immutable source identity, recipe,
and semantic exercise key, not from prompt text, glosses, HTML, tags, or local Anki IDs. The
configured profile tags are emitted in the `Tags` column as Anki's special tags metadata,
not as a regular note field; provenance fields are included for auditing. `Source Path` is
intentionally blank to avoid leaking absolute paths. The profile's language tag is explicit
(`de` for the approved German content); it does not localize CLI controls or diagnostics.
The user-owned `Personal Notes` field is deliberately **not** a CSV column: generated import
data covers only managed fields, so a repeat import can never offer an empty personal value
for accidental overwrite. Generated CSVs and manifests still contain source-derived text,
stable IDs, and provenance; treat them as sensitive and redact them before sharing.

Before the first import, create the dedicated note type named by the profile (normally
`Latinitas Principal Parts`) in Anki. Create regular fields for every `#columns` name except
`Tags`, which is the special tags column, plus a trailing user-owned `Personal Notes` field,
and create at least one template yourself. CSV import headers can preset an existing note
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
3. Map the managed columns (`Prompt`, `Answer`, `Tags`, provenance, and recipe metadata) as
   before. `Personal Notes` stays unmapped because the generated CSV never contains it; no
   remembered Ignore selection is required.
4. Keep **Allow HTML in fields** enabled. Anki's manual documents that matching notes are
   updated in place, remain in their current decks, and preserve scheduling when updating is
   enabled. This behavior was verified in a disposable collection with native Anki Desktop
   26.9.3: repeat imports through freshly opened dialogs updated changed managed content,
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
