# Principal-part preview and deterministic CSV export

The confirmed profile workflow uses the existing `preview` and `generate` entry points
with `--profile`. The legacy USFX corpus path remains available when `--usfx` is supplied.
The profile path never writes the source CSV/APKG/COLPKG, a profile, or an identity manifest
unless an approved ID-less CSV manifest is being persisted beside the generated output.

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
`source.csv.latinitas.json`. The first run must explicitly approve each allocation, for
example:

```bash
uv run latinitas-cards generate \
  --input source.csv \
  --profile .latinitas/profile.json \
  --output generated-principal-parts.csv \
  --approve-allocation 0 \
  --approve-allocation 1
```

Approval row numbers are zero-based source-record indexes. An edited or ambiguous row must
use an explicit `ROW=SOURCE_ID` reuse approval instead. Unresolved reviews block export; no
identity is silently transferred or allocated. The CSV and manifest are staged together; if
either replacement raises a caught process/I/O exception, the exporter attempts to restore
the prior output/manifest pair. If rollback itself fails, retained `.backup.*` files are
reported and recovery is required before retrying; inspect the destinations and backups
instead of blindly retrying or deleting them. The two files are replaced separately, so this
is not durable pair-atomicity: abrupt process termination or power loss between replacements
can leave one file new and the other old. There is no journal or automatic crash recovery;
recover the pair manually before retrying. Use `--manifest` to select a different sidecar path.

## Anki text import

The output is UTF-8 CSV with deterministic `\n` line endings. It uses Anki text-file headers
for `#separator`, `#html`, `#notetype`, `#deck`, `#tags column`, and `#columns`. The first
regular column is always `LatinitasID`; it is derived from immutable source identity, recipe,
and semantic exercise key, not from prompt text, glosses, HTML, tags, or local Anki IDs. The
configured profile tags are emitted in the `Tags` column as Anki's special tags metadata,
not as a regular note field; provenance fields are included for auditing. `Source Path` is
intentionally blank to avoid leaking absolute paths.

Before the first import, create the dedicated note type named by the profile (normally
`Latinitas Principal Parts`) in Anki. Create regular fields for every `#columns` name except
`Tags`, which is the special tags column, and create at least one template yourself. CSV
import headers can preset an existing note type and deck, but they do **not** create note
types, fields, or templates. The `#deck` header selects the target and Anki may create a missing
deck; pre-create it when you
need to establish its hierarchy or settings yourself. The header is not a template/schema
definition. Enable **Allow HTML in fields**, map `Tags` to Anki's tags column rather than to a
note field, and map every other regular column to the corresponding field. Map `Personal Notes`
to its field on the first import.

For repeat imports:

1. Select the same dedicated note type.
2. Keep `LatinitasID` as the first/matching field and choose **Update existing notes when
   first field matches**. Use match scope **note type** (or **note type and deck** when that
   is an intentional local policy).
3. Map the managed columns (`Prompt`, `Answer`, `Tags`, provenance, and recipe metadata) as
   before, and set `Personal Notes` to **Ignore field** so existing personal notes are not
   overwritten.
4. Keep **Allow HTML in fields** enabled. Anki updates matching notes in place and preserves
   their scheduling; an existing note remains in its current deck when updated.

Anki's authoritative text-import behavior is documented in the
[Anki Manual: Text Files](https://docs.ankiweb.net/importing/text-files.html), including
UTF-8, HTML, first-field duplicate matching, update behavior, and supported file headers.
The generated CSV is the auditable file boundary; it does not write a live Anki collection.
