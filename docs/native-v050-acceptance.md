# Planned Linux Anki Desktop acceptance kit

**Status: prepared, native steps unexecuted in T-199.** T-201 must record the
actual candidate, client, settings, outputs and failures. This kit is not native
acceptance, T-195 manual-resolution proof, an accuracy study or publication approval.
It excludes AnkiMobile, sync and existing scheduled user collections.

There are two evidence lanes. Keep their results separate:

1. **Synthetic display/transport controls:** hand-built `display-*.csv` from the
   [example runner](examples/v050_disposable.py) use actual fields, templates and
   presentation functions. They show human/automatic siblings while production
   automatic policy remains disabled. They have no acceptance/export receipts and
   are not pipeline-generated accepted learning cards. Native success on these
   controls cannot accept the original automatic release scope.
2. **Integrated product acceptance:** use actual library resources, permitted
   inputs, exact real human/curated approvals, receipts and supported destination
   workflows. Automatic endpoints and policy cells require their separate T-200
   prerequisites/evaluation/enablement and qualified token accounting. Mark those
   cells blocked until authorized; do not quietly replace them with lane 1 or
   human-reviewed examples. No real inference/reference acquisition is authorized
   by this kit. Record unavailable prerequisites rather than a false pass.

## Prerequisites and disposable setup

Use Linux Desktop with a working display and an independently installed Anki
Desktop client. Record `anki --version`, Python, Qt/WebEngine, OS and architecture.
Prior evidence used 26.09.3, but it does not certify this candidate or client.
Do not install into the normal project environment or reuse a personal profile.
Provision optional native dependencies only for the separately authorized T-201
run. SQLite CLI is required for the closed-database comparisons below.

```sh
# From the exact candidate checkout, after mise run setup:
mkdir -p .amp/in/orb-backup
uv run python docs/examples/v050_disposable.py .amp/in/orb-backup/v050-native-inputs
kit="$PWD/.amp/in/orb-backup/v050-native-inputs"
mkdir "$kit/anki-base"
git rev-parse HEAD > "$kit/candidate.txt"
anki --version > "$kit/client.txt"
# Run only during T-201, with no other Anki process or sync:
anki -b "$kit/anki-base"
```

Create a fresh profile named `T201 disposable`, disable sync, and never point `-b`
at another profile. In an orb, launch the GUI with the supported Desktop/service
workflow; a browser preview is not Anki. Quit gracefully before every backup or SQL
capture. Never copy/read/write its live SQLite database. Keep all inputs, histories,
run directories, receipts and output CSVs with hashes and the exact checkout.

## Install opt-in layouts and import controls

For each `de` and `en`, manually create a **new** note type named `T199 display de`
or `T199 display en`. From `<language>/grammar-migration.json`, install fields in
exact order, all templates in ordinal order, and CSS verbatim. Do not omit unused
templates and thereby renumber ordinals. This is the grammar `acceptance-1` plan;
it includes `Personal Notes`. Confirm the installed field/template/CSS values by
inspection. CSV does not install them. Do not modify any existing note type.

Import `<language>/display-initial.csv` via Desktop's actual Import dialog:

- Comma, HTML enabled, matching the explicit note type, first field `LatinitasID`.
- Map each named column to its field, Tags to the final column, `Personal Notes`
  to `(Nothing)`; leave extra automatic import tags empty.
- Select a dedicated deck `T201::<language>`. For subsequent imports select
  **Update existing notes**, match scope **Note Type**, and inspect every mapping
  again. Capture the actual dialog, not only the CSV or a backend call.

Independent expected inventory **per language**: exactly two notes and three
cards. `T199-display-<language>-mixed` has ordinals 12 and 13; the human-only note
has ordinal 12. No other card is enabled. Mixed note tags include
`latinitas::acceptance::mixed`, not `latinitas::acceptance::human`; human-only note
has the latter. Both retain `User::Keep` and `SourceCase`. The JSON in Acceptance
Data has the two distinct per-card origins for the mixed note. These labels are
synthetic controls, not claims about linguistic validity.

In Browser add unequal `Personal Notes` values (`de mixed keep`, `de human keep`,
and corresponding English values). The controls already include user/source tags
so the complete update Tags column must preserve them. For additional manually
added tags, first review and add those tags to each affected update row; never
import a partial tag set and call it preservation. In lane 2 use fresh destination
capture and explicit full-field/tag review instead of this controlled edit.

## Appearance and interaction matrix

Record each cell for **both languages and both note types/families** used in lane 2.
For lane 1, exercise every mixed/human sibling, not just the note summary. Use
native Browser preview and actual reviewer. Record a short keyboard/pointer video
and screenshots; state whether a step ran in Browser preview or reviewer.

| Cell | Independent expected result |
| --- | --- |
| Light, question | Only the prompt; no answer, stem solution or acceptance/evidence panel leaks onto the front. |
| Light, answer collapsed | Readable answer; `Warum diese Antwort?` / `Why this answer?` summary is visible and evidence is closed initially. |
| Light, answer expanded | Correct per-card human/automatic origin, supporting checks and synthetic limitation; no sibling's origin misrepresented as this card's authority. |
| Dark, collapsed and expanded | Repeat both states; text, border and focus remain distinguishable. Record actual client theme as well as profile presentation configuration. |
| Keyboard | Tab to summary, visible focus; Enter and Space toggle without a mouse; Shift-Tab can leave it. No accidental review grade. Record actual behavior, not inferred HTML accessibility. |
| Pointer | Move over summary, neighboring content and next sibling; click toggles only the chosen panel. No panel on questions. |
| Mixed note | Ordinal 12 human, ordinal 13 automatic; note filter says mixed. Neither means all siblings are automatically qualified. |
| Human-only control | Human label, no automatic label. Switching cards/themes must not retain another card's expanded state or evidence. |

German labels belong in German; English labels in English. Reference translations
are self-assessment guidance, not exact-string grading. For curated lane 2 check
rule-only cards without an invented aid, original German aid preservation, separate
English review, and formation fronts without the tested answer. Theme changes that
alter rendering require fresh exact rendered approval before export; do not reuse a
light-theme approval for a changed dark presentation. Client theme-only toggles and
changed profile configuration must be distinguished in the evidence.

## Closed baseline, update and omission

Before baseline, create a small **disposable** history using actual reviews on
chosen cards and leave another new; record the actions. This setup is not a claim
of scheduler-generated long-term preservation. Stop reviewing after baseline.
Do not use Preview as proof that a review log was created. Quit Desktop, locate
the newly created profile's `collection.anki2`, and make a closed full-profile
backup. Set `collection` to that absolute path, never a user collection.

```sh
# Anki must be closed. Set collection to the disposable profile's actual path.
cp -a "$kit/anki-base" "$kit/baseline-profile"
mkdir "$kit/before" "$kit/after"
for table in cards revlog notes notetypes fields templates decks deck_config; do
  sqlite3 -readonly "$collection" ".dump $table" > "$kit/before/$table.sql"
done
sqlite3 -readonly -json "$collection" \
  'SELECT id,guid,mid,tags,flds FROM notes ORDER BY id' > "$kit/before/identities.json"
```

Stop if a required table is absent on this client. Record the actual native schema
and adapt the capture explicitly; an empty dump is not passing evidence. Save the
full closed database too, not just selected SQL. Hash input/output and installed
layout artifacts. Capture native IDs/GUIDs, first-field values, CardKey fields,
note/card/template bindings, Personal Notes and complete tag sets independently of
the generator's receipt IDs.

Reopen the disposable profile. Import `display-update.csv` in the matching language
note type with the verified update settings. Expected: only mixed note
`Grammar12Answer` acquires prefix `Controlled correction: `. Human-only note and
mixed ordinal 13 answer stay unchanged. Card set and identities do not change.
Repeat the exact update: no new notes/cards or further content changes. Quit, dump
the same tables to `after`, then compare:

```sh
for table in cards revlog notetypes fields templates decks deck_config; do
  sqlite3 -readonly "$collection" ".dump $table" > "$kit/after/$table.sql"
  cmp "$kit/before/$table.sql" "$kit/after/$table.sql" || exit 1
done
sqlite3 -readonly -json "$collection" \
  'SELECT id,guid,mid,tags,flds FROM notes ORDER BY id' > "$kit/after/identities.json"
```

Require **every column of every card and revlog row** identical, not only counts or
due dates. Decode `flds` by native field order (separator U+001F); allow only the
specified answer prefix on that note plus native note `mod`/`usn` bookkeeping.
Check all other note columns separately, exact Personal Notes and complete tags.
Record full before/after comparisons; a matching receipt or import success banner
is insufficient. If GUI startup modifies another captured table, report the exact
delta and investigate rather than widening the allowlist silently.

Import `display-omission.csv` (only human-only row) and capture again. Expected:
mixed note and both cards **remain installed unchanged**, including its now-stale
answer. Omission never retires/suspends anything. This deliberate control establishes
why a separate manual handoff is required, not a real policy withdrawal.

## Integrated withdrawal and manual resolution (T-201/T-195)

For lane 2 use a currently eligible real human/curated export and retained history,
not the example runner's deliberately replaced record. Install the matching grammar
or curated note type manually and attest it with `pipeline confirm-layout` only
after actual installation. Curated uses its own family plan and note type, not
grammar fields. Export with exact current approvals; save the receipt and import.
Independently capture the same native baseline, including related recipe siblings.

For curated, submit an explicit current rejection superseding the prior approval
via `curated review`, retain old history and regenerate with receipt reconciliation.
For policy withdrawal, **do not enable or withdraw production policy for this kit**:
that native scenario stays gated until exact authorized policy revisions exist.
Synthetic isolated withdrawal tests are engineering evidence only.

### Source correction against installed material (planned)

Start from lane 2's genuinely reviewed, imported export, not `display-*.csv` or
the runner's replaced record. Keep the run at its original absolute path. Retain
every ancestor receipt/CSV, source, profile, extraction, analysis, stage and decision;
save the last receipt ID as `prior_receipt`. Independently inventory each installed
`LatinitasID`, native note/card ID, template/card key and prior decision ID across
all batches. Choose a source entry with an installed card, a still-current eligible
recipe sibling on the same note, and an unrelated eligible note as controls. If
that baseline is unavailable, mark the sibling/control cells blocked, not passed.
Capture the closed-Desktop baseline and Personal Notes/tags as above.

Using the already permitted source and its actual review/authoring process, correct
one erroneous extraction assertion for that entry; corroborate it against the
mapped source cell. If the error is in the source cell, correct that source too.
Keep source identity/scope, objective keys, targets and recipe slots unchanged;
changing those is a separate new-identity case. Preserve the original files and
record the exact old/new text, evidence and hashes. Recreate the reviewed v2 input
binding with the library's `pipeline_consumers.input_binding(data, deck,
source_bytes, scope)` and explicitly confirm a new pipeline profile's input/deck
SHA-256 references. These are commitments to reviewed material, not new approval.
There is no CLI that automatically rebinds or approves corrected extraction.
If this genuine authoring/review prerequisite cannot be met, block the scenario;
do not substitute arbitrary JSON edits, test `rebind`, or synthetic authorities.

Set `installed_run` to the retained run and `renewed_profile` to that confirmed
profile. From the candidate checkout, before obtaining any fresh acceptance:

```sh
uv run latinitas-cards pipeline validate --profile "$renewed_profile" --json
uv run latinitas-cards pipeline preview --run "$installed_run" --profile "$renewed_profile" --json
uv run latinitas-cards pipeline handoff --run "$installed_run" --profile "$renewed_profile" --json
# Negative control: changed eligibility must not silently continue the old batch.
uv run latinitas-cards pipeline export --run "$installed_run" --profile "$renewed_profile" \
  --output source-unsafe-continuation.csv --after-receipt "$prior_receipt" --json
# Expect nonzero reconciliation-required refusal and no continuation CSV.
uv run latinitas-cards pipeline export --run "$installed_run" --profile "$renewed_profile" \
  --output source-reconciled.csv --reconcile-receipt "$prior_receipt" --json
```

Independent expectations: old acceptance cannot approve the changed assertion or
binding; preview withholds affected cards. Handoff/reconciliation names the exact
previously installed note/card and prior decision IDs, with the actual reason
(`assertion_mismatch` for changed assertions; a binding-only change may differ),
manual resolution and `not_applied`. No stale row is refreshed. Any unresolved
installed sibling withholds the **whole note row**, including an otherwise eligible
sibling; independently eligible unrelated notes remain eligible, subject to normal
selection and already-emitted receipt handling. They need not be emitted again.
Compare identities and row membership, not a fixed fixture count. Do not erase old
decisions or re-approve them merely to make this negative control export.

After recording the refusal, use offline resume for an **orchestrated** run so it
refreshes ingest before analysis and later prerequisites:

```sh
uv run latinitas-cards pipeline resume --run "$installed_run" \
  --profile "$renewed_profile" --offline --json
```

Inspect every stage outcome; callback success alone does not mean regenerated
analysis or acceptance. Missing resources or authority can still defer/withhold.
An **adapted** run has no standalone ingest/analyze executor: genuinely regenerate
current analysis and bound stages through the authorized library/authoring process
first, then use `pipeline stage decide --run "$installed_run" --profile
"$renewed_profile" --offline --json`. That command consumes analysis; it does not
refresh it and rejects stale context/assertions. If current analysis cannot be
produced, regeneration remains blocked. Obtain genuinely fresh full-card review
through the supported review path before any renewed learning export; offline
stages alone do not grant acceptance. Re-run preview/handoff and explicit receipt reconciliation.
Fresh acceptance does not itself resolve the old installed-material obligation:
retain its history and use the content-only manual resolution procedure below.

### Application-checker revision against all installed batches (planned/blocked)

Require two maintainer-provided application revisions with an actual reviewed
checker/validator implementation change. Record each checkout/package identity,
imported module path and `acceptance_policy.implementation_digest()`; the digests
must genuinely differ. The digest covers application Python dependencies, not a
profile's display theme or a hand-edited version label. There is no CLI switch to
install a checker revision. Do not edit stored acceptance/version/digest records,
monkeypatch the checker, or treat the isolated test's synthetic revision as native
qualification. If a real revision pair or genuinely accepted installed baseline
is unavailable, this cell is **blocked**. Any automatic baseline additionally
requires the separately authorized exact policy scope/revisions and T-200 gates;
the disabled bootstrap cannot supply it. No such revision or scope is enabled here.

Under the old revision, retain at least two imported batches and independently
inventory their installed note/card/decision identities as above. Keep Desktop
closed and the original run, profiles and complete receipt chain at their original
paths. Switch only the executing application to the real new revision using its
approved setup; do not migrate or overwrite acceptance history. Before fresh
analysis/review, run from that revision:

```sh
uv run latinitas-cards pipeline preview --run "$installed_run" --json
uv run latinitas-cards pipeline handoff --run "$installed_run" --json
uv run latinitas-cards pipeline export --run "$installed_run" \
  --output checker-reconciled.csv --reconcile-receipt "$prior_receipt" --json
```

Expect stale decisions to fail current version binding and handoff to cover **all
affected earlier batches**, not just `prior_receipt`: compare the exact union of
installed note/card identities and old decision IDs. With otherwise unchanged
inputs this is `binding_mismatch`; no card dependent on that stale checker approval
may export. Entire unresolved note rows, including eligible siblings, are withheld.
Unlike the single-source case, an unrelated source note whose approval binds the
same old application checker is also ineligible. Only independently current
acceptance can remain eligible; unrelated native notes are never changed by CSV
generation. Regenerate stages and obtain fresh full-card authority under the new
revision as in the source scenario, then reconcile explicitly, preserving the
earlier manual obligations. Missing/corrupt ancestor receipts or CSV hashes must
block reconciliation, not be bypassed by deleting history or starting a clean run.

For **both** scenarios, do not import reconciled output as proof of resolution.
Perform the explicit content-only manual action below, then repeat the closed
before/after identity, every-column card/revlog, Personal Notes and complete tag
comparisons for affected notes, retained siblings and unrelated controls. CSV
generation alone leaves all native state unchanged; omission is not retirement.
Record setup, regeneration/refusal outputs and native actions separately. These
procedures are planned, not executed, and do not authorize source acquisition,
real inference, policy enablement or native work within T-199.

Expected: affected exact note/card identities, previous decision IDs, reason and
manual action are in `pipeline handoff` or curated receipt ineligibility; ordinary
CSV omits unsafe rows but does not alter Anki. Unaffected independently eligible
notes may still export; an unsafe sibling can withhold an entire note row. Locate
the installed item by first-field identity and confirm card/template binding, not
visible lemma alone. In the disposable profile, explicitly review a **content-only
manual correction** removing the stale assertion/status while preserving enabled
slots and all unowned fields/tags. Record before/after content, the human action,
reason, current evidence and unresolved obligations. There is no CLI that attests
this native resolution automatically.

After closing Desktop, compare all card/revlog columns and every unaffected field,
tag and binding again. Expect no scheduling/history change for this content-only
operation. If the chosen action is suspension, record its intentional queue change
separately; do not call that zero scheduling change. Deletion/retirement/new slots
are not retained-field updates. No supported managed grammar/curated adapter exists;
exercise supported managed layouts separately with fresh capture/adoption/approval
and actual observation, following [the existing Desktop recipe](managed-csv-desktop-verification.md).
Its historical result does not prove this candidate's T-195/T-201 resolution.

## Rollback and evidence disposition

With Desktop closed, restore the full disposable baseline profile into a separate
base directory, open it, verify the restored identities and table content, and
reacquire destination evidence before reconciliation/reimport. Do not overwrite an
open profile or infer rollback from restoring an approval JSON file.

For each scenario record `passed`, `failed`, `blocked` or `not_run`, candidate and
artifact hashes, client identity, actual command/dialog settings, independent
expectations, observed deltas and media. Bind videos to native window/client and
candidate; browser captures do not count as native. Any unexpected identity,
Personal Notes, tags, card set, scheduling or history change blocks the corresponding
acceptance. Final high-effort QA and the German-card native video remain outer work;
T-199 prepares this kit and runs no native steps.
