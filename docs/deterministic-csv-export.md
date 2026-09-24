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

## Anki text import

The output is UTF-8 CSV with deterministic `\n` line endings. It uses Anki text-file headers
for `#separator`, `#html`, `#notetype`, `#deck`, `#tags column`, and `#columns`. The first
regular column is always `LatinitasID`; it is derived from immutable source identity, recipe,
and semantic exercise key, not from prompt text, glosses, HTML, tags, or local Anki IDs. The
configured profile tags are emitted in the `Tags` column as Anki's special tags metadata,
not as a regular note field; provenance fields are included for auditing. `Source Path` is
intentionally blank to avoid leaking absolute paths. The profile's language tag is explicit
(`de` for the approved German content); it does not localize CLI controls or diagnostics.
Generated CSVs and manifests still contain source-derived text, stable IDs, and provenance;
treat them as sensitive and redact them before sharing.

Before the first import, create the dedicated note type named by the profile (normally
`Latinitas Principal Parts`) in Anki. Create regular fields for every `#columns` name except
`Tags`, which is the special tags column, and create at least one template yourself. CSV
import headers can preset an existing note type and deck, but they do **not** create note
types, fields, or templates. The `#deck` header selects or presets the target. The current
Anki manual documents that header as presetting an existing deck and documents missing-deck
creation for a deck column; a missing target may still be created in some import flows, but
do not rely on that when hierarchy or settings matter—pre-create it. The header is not a
template/schema definition. Enable **Allow HTML in fields**, map `Tags` to
Anki's tags column rather than to a note field, and map every other regular column to the
corresponding field. Map `Personal Notes` to its field on the first import.

For repeat imports:

1. Select the same dedicated note type.
2. Keep `LatinitasID` as the first/matching field and choose **Update existing notes when
   first field matches**. Use match scope **note type** (or **note type and deck** when that
   is an intentional local policy).
3. Map the managed columns (`Prompt`, `Answer`, `Tags`, provenance, and recipe metadata) as
   before, and set `Personal Notes` to **Ignore field** so existing personal notes are not
   overwritten.
4. Keep **Allow HTML in fields** enabled. Anki's manual documents that matching notes are
   updated in place, remain in their current decks, and preserve scheduling when updating is
   enabled. This repository does not run the native Anki client or a live-collection
   operation; verify those behaviors in a disposable collection before relying on them.

Anki's authoritative text-import behavior is documented in the
[Anki Manual: Text Files](https://docs.ankiweb.net/importing/text-files.html), including
UTF-8, HTML, first-field duplicate matching, update behavior, and supported file headers.
The generated CSV is the auditable file boundary; it does not write a live Anki collection.
