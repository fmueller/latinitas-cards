# Managed CSV native backend verification

This is the T-061 sanitized evidence gate for the conservative v0.2.0 managed
CSV handoff, not certification of arbitrary collections, the Desktop import
dialog, AnkiMobile, or structural transports. The executable recipe is
`scripts/check-managed-anki.py`. It exercises production planning, explicit
subset approval, pending emission, native import, closed destination capture,
observation and recovery; no mocked importer is used.

The [Desktop dialog gate](managed-csv-desktop-verification.md) separately exercises
actual Desktop 26.09.3 controls and reports its exact settings, tag-only outcomes,
native screenshots and complete card/review-log comparisons. API results below
are not substitutes for that GUI evidence.

## Reproduction and fixture

```bash
work=$(mktemp -d)
uv run --with anki==26.9.3 python scripts/check-managed-anki.py \
  --output-dir "$work/evidence"
```

The output directory must not already exist. The script never opens an existing
user collection. No Anki GUI, reviews, sync or other writer runs on these
collections. It closes the dedicated native backend before opening SQLite,
copying/restoring the database or comparing tables. SQLite connections are
explicitly closed before native reopen. This fixture needs no desktop display.
The prior authored gate uses the same optional native backend dependency; it
does not supply managed multi-card proof itself.

The fixture uses the current reference fields, ten frozen templates and CSS,
schema 3, both completion and recognition recipes, present/perfect roles, two
equal visible lemmas with distinct immutable object keys and LatinitasIDs. Native
first import creates two notes and eight cards (ordinals 0, 2, 5, 7 per note).
Closed fixture seeding adds unequal synthetic review schedules, 15 review-log
rows, one user-suspended card, distinct nonempty Personal Notes, a kept lemma and
manual/source tags. These schedules and historical logs are deliberately
synthetic, not claims about scheduler-generated transitions. New/review/interday
sibling burying is set and reacquired through the native config API; all decoded
settings and full config table bytes are retained. We inspect sibling bindings
and configured burying, not simulate scheduler behavior.

Native schema capture derives field names/order, template name/ordinal/front/
back hashes and CSS hash from the actual model, not merely the desired contract.
The recoverable backup is a closed full database copy, not a fake COLPKG; the
fixture has no media. A restoration is actually performed, checked against all
captured tables, reopened and re-imported after journal reconciliation.

## Exact tested transport and settings

Tested on Linux with PyPI `anki==26.9.3`, using Anki's supported native Python
client `Collection.get_csv_metadata` and `Collection.import_csv` with
`ImportCsvRequest`. The backend is the real Anki Rust import implementation, not
a project-defined importer. Fresh metadata is obtained for each file, comma
delimiter, HTML enabled, first-field `LatinitasID`, `CsvMetadata.UPDATE`,
`CsvMetadata.NOTETYPE`, dedicated existing note type and deck, tags column 5,
and final Personal Notes field mapping 0 (unmapped). Complete protobuf metadata
and native result logs are recorded per import; protobuf JSON omits default enum
values, so UPDATE/NOTETYPE are explicitly named here and assigned in the script.

The mixed-outcome fault intentionally changes **only** tags mapping to 0 for one
import. It is not a supported mapping and is never used to claim full success.
The local result-persistence fault is injected after a genuine native import,
before the journal replacement. Native destination data is never mocked.

## Observations and comparisons

| Scenario | Native observation and application result |
|---|---|
| Asymmetric approved subset | Only one note's Meaning and selected source-tag addition change. Its kept lemma and every slot field remain at destination values; the other proposed note is not imported. Manual tag and both Personal Notes survive. |
| Full preservation | All columns of every surviving card and review-log row are identical, including IDs, note/deck/ordinal bindings, queue −1 user suspension, due, interval, factor, reps, lapses, flags and card data. Note-type, fields, templates, decks and deck-config tables remain identical. |
| No-op reapplication | Re-importing the same emitted CSV yields identical full notes/cards/revlog and setup tables; observation is idempotent and the saved journal stays equal. No duplicate notes/cards. |
| Tag-only addition | `manual source` → exact set `manual new source`; every managed field remains unchanged. |
| Tag-only removal | `manual source` → exact set `manual`; every managed field remains unchanged. |
| Final tag removal | `source` → empty set; every managed field remains unchanged. No dummy content or metadata edits force any tag probe. |
| Native import before result-persistence failure | Injected OSError leaves the pending journal byte-identical. Reacquisition of native results and observation confirm only the selected effects, without replaying the import. |
| Partial import | A two-note emission imports only the first approved row. Only its anchor advances; the other remains unchanged/unresolved. Reconciliation and renewed approval emit only the remaining row, then native retry confirms it. |
| Mixed field/tag result | An intentionally unmapped Tags column applies Meaning but not the approved addition. The whole note's selected effects remain unresolved; neither baseline field nor tag ownership is falsely promoted. |
| Changed handoff | A real closed destination tag edit after emission makes approval verification fail as stale. The emitted file is not imported. |
| Backup restoration after recorded success | Full tables are restored. The old recorded-success observation is rejected as inconsistent; explicit reconciliation abandons the old plan, followed by new approval and successful native retry. |
| Structural boundaries | New-slot eligibility, guard clearing, front clearing, changed slot content, retirement and identity consolidation cannot authorize structural/slot writes. Permitted content-only subsets retain actual slot fields and are imported natively with identical actual card sets. |
| Enabled slot / missing actual card | A real card deletion is detected as card-evidence drift. Even after explicit adoption of that inventory, emission of a content subset fails its actual-set guard, leaves the journal byte-identical and publishes no CSV. No implicit native recreation is permitted. |
| Reactivation refusal | The real user-suspended row plus explicitly synthetic historical user-owned retirement provenance yields an unsupported reactivation operation; approval refuses it and actual tables stay identical. No claim of native structural retirement/reactivation is made. |
| Previously anchored missing note | Closed native inventory with one note/cards/logs removed yields a reconciliation conflict with no operations/card effects. Empty approval has no targets/import rows; all remaining destination tables stay identical. |
| Never-anchored missing member | A fresh baseline adopts only the remaining actual members of that same complete inventory. The missing member yields an actual create operation, explicitly selected and refused at approval; all destination tables stay identical. |
| CSS mismatch | Changing the actual native model CSS makes acquired schema evidence incompatible and is refused. No template/style migration is attempted. |
| Separate fresh start | Explicit T-057 destination/schedule approval precedes import into a second collection. The original full database hash and all captured tables are unchanged. The second collection has eight new cards, zero reviews/reps and no inherited suspension, Personal Notes or scheduling. |

For content/tag changes the comparison enumerates exact field/tag deltas and
allows only the affected note's native `mod`/`usn` metadata to change. Every other
note column is checked, including GUID, note-type ID, first-field checksum,
flags/data and Personal Notes. No-op comparison allows no metadata churn. Exact
tag expectations are derived independently from reviewed source ownership and
manual retention; they are not inferred from the importer's return message.

The evidence output includes each closed table capture, schema-bound snapshot,
plan/approval, emitted CSV, native settings/result, full backup and final
collection. `report.json` binds these by SHA-256 and records Python/OS/backend
version, original source commit, script hash and package source hashes. A clean
repeat on a new disposable directory is the release-candidate retest recipe;
source hashes, not a bare task status, define the tested implementation.

### Executed run: 2026-10-05

The reviewed run used Python 3.14.2, Anki 26.9.3 and Linux x86_64/glibc 2.36,
against the application sources at
[`f3864ee`](https://github.com/fmueller/latinitas-cards/commit/f3864ee4de735d6ab2d24dff83153641f5c75ab5).
All 16 top-level scenarios passed. Content/tag import changed exactly the chosen
note's Meaning and tags; no other note columns changed in that capture. Every
captured card/revlog/setup row stayed identical. The missing-card scenario
matched the exact original card-ID set minus the deliberately deleted card,
then retained every captured destination table through the rejected emission.

| Binding | SHA-256 |
|---|---|
| Executed script | `bd3e01b25c6c1b978c797743af89a9101e32f6152fb09986f630fb88b00b4098` |
| Native report | `5b63201e917ef619ae8f65812375afcc28c3afed748dc6dd776e8c914654df85` |
| Recoverable closed backup | `cf388c84e861fce1cf33f34df454723758dca93d2edc4e4c8febd461d550f62d` |

These hashes bind the recorded run, not deterministic collection bytes across
reruns: Anki allocates fresh local IDs and modification metadata each time.

## Capabilities and limitations

- These native API settings demonstrate compatible **existing-note content and
  reconciled full-tag-set updates**, including isolated tag-only addition,
  removal and final removal, for this version/fixture. This does not erase the
  older Desktop-dialog evidence of tag-only skips. Do not generalize native API
  results to the GUI's settings; verify the chosen actual path, and keep any
  skipped effects pending/unresolved or mark them unsupported.
- Slot prompt/answer/guard changes remain unsupported, even if a raw importer
  could change them. Creation, retirement, reactivation, consolidation, APKG
  managed updates and CSS/template migration remain unsupported. Fresh import
  is a separately approved new destination with new scheduling, not migration.
- The file transport cannot lock or observe an intervening GUI edit. Fresh
  acquisition and a genuinely edit-free interval remain operator obligations;
  artifact emission, import-dialog success and equal later values are not proof.
  The CLI's `native_safety` report describes offline emission, manual native handoff
  and observed application; it does not dynamically certify a client or import
  a collection itself. Backup recoverability and freshness remain operator attestations.
- No rendering, Desktop dialog interaction, AnkiMobile presentation, sync,
  review-session scheduler or real user-data compatibility is certified here.
  T-062 remains the separate mobile gate. T-060 owns affected recipe retests
  after later form-parsing changes. Repeat affected native checks whenever
  managed content, schema, templates, CSS, transport or dependency settings
  change for a release candidate.
