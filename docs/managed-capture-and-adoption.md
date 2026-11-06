# Closed-backup capture and first adoption

`managed capture` produces the existing validated version-1 destination snapshot;
`managed adopt` produces the existing version-1 observed baseline. These are
acquisition and ownership review, **not permission to apply an update**. Existing
plan/approve/emit/observe/reconcile gates and structural restrictions are unchanged.

## Backup and binding prerequisites

1. Stop reviews, edits and sync on **every device** for the entire capture/import/
   observation interval. Obtain a fresh, recoverable full collection backup including
   schedules, review logs and settings. Close Anki. Keep the untouched backup.
2. Use only a disposable **closed, checkpointed SQLite backup copy**, never a running
   user's `collection.anki2`. Capture rejects `-wal`, `-shm` and `-journal` sidecars;
   do not delete them to force success. Obtain a properly closed complete copy instead.
   `.apkg`/`.colpkg` archives, schema-11 backups and collections with filtered decks
   are not accepted by this command.
   It does not unpack, migrate, provision, or open a collection through Anki.
3. Record the operator-assigned original collection and profile bindings. A path,
   deck name, model ID or `col.id` is not a portable collection identity. Rebinding a
   copied/restored collection requires separate reconciliation, not first adoption
   over an existing journal.
4. Supply the dedicated compatible Latinitas note-type ID and **whole managed-set
   membership**, including retired and currently absent source/object identities.
   Use immutable provenance from the generation source, never infer identity from
   visible Latin text. The command derives the actual schema from stored field names/
   order, template names/ordinals/front/back and CSS and compares it with the frozen
   registry/reference contract. Different layouts require separately reviewed manual
   setup; no scheduled templates are rebound.

The supported storage layout is Anki SQLite schema 18, tested with `anki==26.9.3`.
Its native model/template configs are protobuf blobs. The optional pinned package
supplies generated protobuf definitions only; capture uses read-only, immutable
SQLite and never constructs `Collection` or invokes the native backend. Other commands
and the default installation do not require Anki. No guess at native collations or
storage formats is substituted for missing definitions.

## Selection and capture

`selection.json` contains exactly the binding inputs below. This is an operator
assertion of the original destination and whole immutable membership, not proof that
the backup is from that destination. The native note-type ID must be a string.

```json
{
  "destination": "my-original-latin-collection",
  "profile": "my-confirmed-profile",
  "note_type_id": "native numeric ID as a string",
  "managed_set": {
    "scope": "my-immutable-source-scope",
    "members": [
      ["generated LatinitasID", "my-immutable-source-scope", "immutable source ID", "immutable object key"]
    ]
  }
}
```

Use real generated identities, not these placeholders. Membership uses the existing
snapshot contract; keyed members must match `derive_latinitas_id`. Capture scans the
entire dedicated note type: unknown/missing/duplicate IDs, identities under a different
model, missing columns/tables, incompatible schema and ambiguous card bindings fail
closed. Absent selected identities remain in membership but not in observed notes.
Actual stored cards and logs are captured; enabled fields never manufacture cards.
Suspended cards are observations only, not evidence of tool-owned suspension.

```bash
uv run --with anki==26.9.3 latinitas-cards managed capture closed-backup.anki2 \
  --selection selection.json --closed-backup --interval-confirmed > snapshot.json
```

Both flags are required attestations. `--closed-backup` attests a closed backup copy.
`--interval-confirmed` attests a fresh complete backup and no intervening reviews,
edits or sync on any device since it was made. SQL reads and the hash use the same
private temporary byte copy. Capture checks source file identity/size/timestamps and
sidecar absence before/after acquisition; temporary copies are removed on exit. These
checks are not a lock or proof that no writer briefly ran. Neither a SHA-256 nor a new
capture timestamp proves freshness or original collection identity. If either assertion
is false or uncertain, do not pass the flags; recapture/replan instead. No TTL makes a
stale backup safe.

For `managed emit`, the backup prerequisite is **operator-attested recoverability**:
the command checks for a nonempty file, prevents journal/output aliases, and records
its SHA-256 plus your recovery instructions. It does not test restoration, validate
the file as a native backup, or prove that schedules/history/settings are recoverable.
Keep a real full backup and verify your recovery procedure independently. Snapshot
`fresh: true` and interval confirmations are assertions, not age-bound enforcement.
Closed-backup capture's boundary checks do not lock the original collection.

Managed field values reject active C0/C1 controls (including NUL, ESC, the native
field separator U+001F, and DEL), except tab, CR and LF. Supported multiline and
non-ASCII text is retained exactly in the offline CSV/journal; this is not a claim
that a native importer never normalizes text. Terminal JSON escaping is a separate
presentation safeguard, not CSV sanitization. Personal Notes are outside managed
validation/writes and are never sanitized or mutated by this policy.

Output is local sensitive evidence: `export_options.tables` retains every column of
`col`, `config`, notes, cards, revlog, notetypes, fields, templates, decks and deck_config;
binary configs are hex encoded without dropping settings. It includes raw personal
text and unrelated collection rows for local comparison, so **do not publish it**.
The managed note values and adopted baseline exclude Personal Notes; only its digest
is retained with the adopted note. Deck/deck-config bytes bind settings. Mutable
collection timestamps and collection config are captured separately in table evidence,
not misclassified as deck options that must stay identical through a content import.

## Explicit ownership review and adoption

Review every observed note before writing `ownership.json`. Map each portable identity
to its separate source-inherited/configured contributions and explicit keep overrides:

```json
{
  "generated LatinitasID": {
    "source_tags": ["source"],
    "configured_tags": [],
    "keep_tags": ["manual"],
    "keep_fields": []
  }
}
```

Every observed tag must be accounted for. Overlapping source/configured origins are
kept separately; a managed-looking namespace does not prove ownership. The existing
contract also supports explicitly reviewed `lifecycle_tags` and `suppressed_tags`.
Unknown/lifecycle-colliding ownership must be resolved by the reviewer; no suspension
or reactivation is authorized. Do not put Personal Notes in `keep_fields` or managed
values. The review record must describe the per-note ownership review, not merely
generation/profile confirmation.

```bash
uv run latinitas-cards managed adopt snapshot.json --ownership ownership.json \
  --review 'reviewed each listed note and its origins/keep overrides' --state baseline.json
```

The journal must be new; existing files (including corrupt/pending ones) are never
overwritten or resumed. Use `managed reconcile` for reviewed recovery and replan with
new apply approval. Capture/adoption does not import a CSV or advance pending effects.

## Refreshing German status text on imported notes

The German status-text correction changes presentation, not linguistic evidence,
LatinitasID, semantic card keys, eligibility or scheduling. Regenerate with the
same source identities, confirmed profile, recipes and claim reviews; do not rerun
setup or change the note type. Hidden `source-extraction` payloads may still contain
English/JSON. Do not translate them or free-text source/reviewer evidence.

For an already imported note, use the existing observed `baseline.json`; if none
exists, capture and explicitly adopt the destination as described above first.
The exact request/envelope formats are in the
[file-transport guide](destination-snapshots-and-file-transport.md#managed-json-review-and-approval).

1. Back up the full collection, verify recoverability, stop reviews/edits/sync on
   every device, and capture a fresh closed copy with the original selection:

   ```bash
   uv run --with anki==26.9.3 latinitas-cards managed capture before.anki2 \
     --selection selection.json --closed-backup --interval-confirmed > before.json
   ```

2. Build `request.json` with the original binding, `before.json` as `snapshot`, the
   current journal as `baseline`, and the full unchanged `effective_profile`
   (including `generated_note_type` and `target_deck`). Select the regenerated notes
   as proposals using their complete managed fields, excluding Tags and Personal
   Notes; preserve reviewed tag contributions. Run:

   ```bash
   uv run latinitas-cards managed plan request.json > plan.json
   ```

   Review the actual diff. Only translated `Principal Parts` and affected
   `Completion*Prompt`, `Completion*Answer`, `Recognition*Answer` (or reference
   presentation) fields should change. No create/retire/reactivate, identity,
   eligibility, schema or card-set effect is part of this fix. Stop if any appear;
   do not approve unsupported effects to force an import. Destination edits and keep
   overrides still require their normal conflict review.
3. Approve the exact supported field operation IDs from `plan.json`, repeating
   `--operation` for each selected translation change. Presentation changes need
   new apply approval, not renewed linguistic approval:

   ```bash
   uv run latinitas-cards managed approve plan.json \
     --operation '<exact field operation ID from plan.json>' \
     --review 'German status translation only; retained identities and cards' > approval.json
   ```

4. Assemble `handoff.json` from `binding`, fresh `snapshot`, `plan`, `approval`,
   recoverable `backup` path, verified `recovery` instructions, `note_type` and
   `deck`, then emit once:

   ```bash
   uv run latinitas-cards managed emit handoff.json --state baseline.json --output updates.csv
   ```

   In the bound Anki collection, import as **Update**, matching first-field
   LatinitasID + Note Type and the emitted column mapping. Leave Personal Notes
   unmapped. Do not delete/reimport notes or add cards. Retain the native report.
5. Before any reviews/edits/sync, close Anki and capture the post-import closed copy
   using the same selection and capture flags, writing `after.json`. Prepare
   `observation.json` with `binding`, that snapshot, `plan_id` and the native `report`:

   ```bash
   uv run latinitas-cards managed observe observation.json --state baseline.json --interval-confirmed
   ```

   Check that every selected operation is observed and none remains unresolved.
   Observation compares unchanged card/history and other preservation evidence;
   emission alone does not prove scheduling preservation. Native client/settings
   limits in the full file-transport checklist still apply.
6. **Reconcile only for recovery**, not as an unconditional final step. For partial,
   unknown or restored outcomes, reacquire a fresh snapshot and review ownership of
   every observed note. Write `reconciliation.json` with `binding`, `snapshot`,
   `ownership` (the adoption format) and an explicit `review`, then run:

   ```bash
   uv run latinitas-cards managed reconcile reconciliation.json --state baseline.json
   ```

   This accepts current anchors and abandons pending effects; it is not proof that
   the update succeeded. Replan and obtain new approval for remaining writes.
   Never replay the old CSV blindly. On complete successful observation, no
   reconciliation or new scheduling is needed.

## Executable sanitized walkthrough

From the checkout, run the README recipe:

```bash
work=$(mktemp -d)
uv run --with anki==26.9.3 python scripts/check-managed-capture.py \
  --output-dir "$work/evidence"
```

The new output directory contains generated selection/ownership/request/handoff/
observation files, snapshots and the baseline, CSV and native import report.
`commands.json` records the exact installed CLI invocations. The recipe reuses the
existing native fixture provisioning/import code, takes only its own disposable paths,
and uses production capture/adoption for both backups. It checks baseline advancement
only after observation, full surviving card/log/model/deck rows, Personal Notes and
manual tags. No user collection is opened. This is native API evidence, **not a Desktop
dialog, AnkiMobile, or general structural-transport claim**. For real imports follow
the [full file-transport checklist](destination-snapshots-and-file-transport.md).
