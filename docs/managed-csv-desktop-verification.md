# Managed CSV Desktop dialog verification

This sanitized gate exercises the **actual Anki Desktop CSV import dialog**, not
the [separate backend API proof](managed-csv-native-verification.md). It certifies
only the fixture, client and settings below. It does not certify arbitrary user
collections, Mobile, sync, structural changes or scheduler transitions.

## Client and actual interaction path

The executed Linux x86_64/glibc 2.36 run used Desktop **26.09.3** (`aqt` and `anki`
26.9.3), Python 3.14.2, PyQt6 6.11.0 and Qt 6.11.2, on orb Desktop/Xwayland.
The optional script `scripts/check-managed-desktop.py` opens Anki with an isolated
profile and calls its normal `aqt.import_export.importing.import_file` entry point.
That opens the native `ImportDialog` and its Qt WebEngine CSV page. CDP connects
to **that webview**, not a Chrome copy of the page or a project preview.

`--drive` clicks the rendered Svelte controls for the note type, Update, Note Type
match scope, every field mapping and Tags, and then clicks the visible **Import**
button. It never invokes `Collection.import_csv`, a frontend import function or
a backend HTTP request for the tested update. Comma, HTML and the selected deck
are checked in the dialog; CSV directives force comma/HTML, so those controls
are disabled. Screenshots capture the actual native Qt dialog's client surface
with `grab()`. GUI closure is graceful before copying or inspecting SQLite.

| Setting | Verified value |
|---|---|
| Field separator / HTML | Comma / enabled, both forced by emitted CSV |
| Note type / deck | `Latinitas Native Managed` / `Latin::Managed` |
| Existing notes / match scope | Update / Note Type |
| First field | `LatinitasID` ← column 1 |
| Managed content and slots | Matching named columns; emitted destination values retained for unapproved fields |
| Personal Notes | `(Nothing)` — no importable write |
| Tags | Column 5, reconciled complete set |
| Tag all notes / Tag updated notes | Empty |

## Fixture and decisive comparisons

Setup reuses `scripts/check-managed-anki.py`, not a new provisioning/import policy.
The two distinct immutable identities share visible lemmas, have eight unequal
scheduled sibling cards (ordinals 0, 2, 5, 7), fifteen synthetic review logs, one
user-suspended card, manual tags and distinct personal text. Bury-new/review/
interday settings are enabled. Final removal deliberately starts one fixture
note with only `source`; the other note still has its manual tag.

The current production closed-backup `capture_backup` and reviewed adoption
workflow bind the plan, explicit subset approval, emission and actual observation.
Only the first captured identity is approved; a proposed lemma overwrite and the
other note's Meaning overwrite are excluded. No dummy field, generator metadata
or content edit forces tag-only imports.

| Dialog scenario | Exact destination outcome | Managed observation |
|---|---|---|
| Content + tag subset | Only chosen Meaning → `approved Desktop A`; `manual source` → `added manual source` | Both effects observed; pending/unresolved empty |
| Tag-only addition | `manual source` → `added manual source`; every field unchanged | Tag effect observed |
| Tag-only removal | `manual source` → `manual`; every field unchanged | Tag effect observed |
| Final removal | `source` → empty; every field unchanged | Tag effect observed |
| No-op reapplication | Same CSV reapplied to the closed result of the actual Desktop content import; all compared rows/columns identical | Recorded journal unchanged by repeat observation |
| **Fault: Tags deliberately unmapped** | Dialog says **Skipped**; intended addition absent; every compared row/column unchanged | Pending before import → **unresolved** after observation; no observed effect or anchor advancement |

The last row is an intentionally **unsupported mapping**, not evidence that the
correctly mapped current client skips tag-only changes. Earlier skip evidence
must not override this run: all three correctly mapped tag-only probes applied.
Conversely, a dialog reporting an already-present note is not success when an
approved effect is absent. The real `observe_updates` call classifies that effect
as unresolved (not simultaneously pending) and leaves ownership/baselines intact.

For every scenario the comparison checks **all columns of all eight card rows
and fifteen revlog rows**, not a scheduling subset. Note/card identities, sibling
bindings, suspension, due, interval, factor, reps/lapses, flags and data survive.
All note-type, field, template, deck and deck-config rows are identical. Note
comparison permits only approved Meaning/tags plus the affected note's native
`mod`/`usn`; every other note column and both personal texts are checked. No-op
and skipped import permit no metadata churn. Broader collection config/`col`
tables are captured too, but are not claimed byte-identical across GUI startup.

## Reproduction and evidence binding

Install the optional `aqt==26.9.3` in a disposable environment with this package.
Start native Desktop (`amp orb desktop ensure` in an orb). On Linux inspect
`libqxcb.so` with `ldd`: this run installed missing xkbcommon-x11 and xcb cursor,
util, image, keysyms, render-util and xkb libraries. No user collection is used.

```bash
# Run from the checkout. DIR must not exist; use an absolute path.
work=$(mktemp -d)
python=$(command -v python) # Python in the environment containing aqt and this package
"$python" scripts/check-managed-desktop.py --output-dir "$work/evidence" --prepare

# Set case to each value in order: content, tag-add, tag-remove,
# tag-final-remove, tag-unmapped, noop. Prepare only once.
case=content
amp orb service start "managed-desktop-$case" --command \
  "DISPLAY=:0 QTWEBENGINE_DISABLE_SANDBOX=1 QTWEBENGINE_REMOTE_DEBUGGING=9223 $python $PWD/scripts/check-managed-desktop.py --output-dir $work/evidence --gui $case"
# Wait for the CSV dialog to appear. Anki must own CDP port 9223.
"$python" scripts/check-managed-desktop.py --output-dir "$work/evidence" --drive "$case"
"$python" scripts/check-managed-desktop.py --output-dir "$work/evidence" --finish "$case"
agent-browser --session t076 close
amp orb service stop "managed-desktop-$case"
```

Use the installed environment's absolute Python path in service commands. The
Linux test-only sandbox override is not advice for a real user installation.
Do not run these helpers against a live/user profile. A GUI phase is single-use;
rerun `--prepare` in a **new** directory for a fresh check.

`bindings.json` records client/Qt/Python/OS, base source commit and SHA-256 of the
harness, existing fixture and application sources. Each case retains closed
before/after backups and complete snapshots, the approval/emitted CSV, exact
dialog control values/result text, native file/settings/mapping/result captures,
actual managed report and evidence hashes. Source hashes, not the task status,
bind the tested implementation. Keep runtime artifacts outside committed notes.

For real file handoffs follow the [destination safety checklist](destination-snapshots-and-file-transport.md):
fresh snapshot, backup, no intervening edits through import, then observe. This
fixture does not make CSV able to lock a collection or certify arbitrary clients.
