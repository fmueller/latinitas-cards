# Legacy note-model transitions and fresh starts

This is the T-034 policy for collections that still carry pre-release
Latinitas note models. It defines how those models are detected, which
transition options exist, and which stay read-only. The machine-checked core
of this policy lives in `latinitas_cards.legacy_transition`; this document
consumes it and never redefines it. Nothing here reads, writes, or mutates an
Anki collection.

## What counts as a legacy model

| Destination evidence | Classification |
| --- | --- |
| First field `latinitas-v1-…` with `Note Schema` 1 or 2 (or no schema field) | Legacy per-exercise note |
| `Note Schema` 1 or 2 without a recognizable current identity | Legacy per-exercise note |
| First field `legacy-split-guid-v1-…` (old experimental `split` clones) | Legacy split clone |
| First field `latinitas-v2-…` with `Note Schema` 3 (or no schema field) | Current learning-object note |
| Contradictions (for example `latinitas-v2-…` with `Note Schema` 2, or a split GUID with any schema) | Unrecognized — review only |

The old recipe/role-derived IDs and per-exercise notes are **never silently
reinterpreted**: the current model is identity version `v2` with
`Note Schema` `3`, a `latinitas-v1-` identity is never treated as a
`latinitas-v2-` object identity, legacy IDs are never reused for a different
object, and old note types are never reused for new-model rows. Exporting is
rejected outright (`LegacyTransitionError`) when the effective profile's
generated note type matches a legacy note type you declare on the supported
`preview`/`generate --profile` path with `--legacy-note-type` (repeatable);
the same rejection is raised by `plan_legacy_transition` for library callers.
Latinitas has no destination awareness in v0.1.0, so it cannot detect legacy
note types on its own: declare them from the inventory in step 2 below.

## The four decisions

| Decision | Meaning | Writes old data? |
| --- | --- | --- |
| `compatible_regeneration` | Current learning-object note: regeneration re-exports the unchanged `LatinitasID`, so Anki's first-field match updates managed fields in place; note identity, surviving cards, and review history are preserved, and eligibility loss is withheld by the prior-export checkpoint instead of deleting cards | No — updates managed fields of existing current notes |
| `legacy_review_required` | Legacy or unrecognized note without a complete explicit approval: a read-only review outcome listing every missing confirmation | No |
| `retain_and_defer` | Legacy note with valuable review history or a conflicting personal annotation: keep the old collection and defer conversion | No |
| `fresh_start` | The only supported legacy option: an explicitly approved, backed-up fresh start of **disposable** data into a new dedicated note type with **new identities and new schedules** | No — old collection stays intact |

No decision reconciles personal annotations or tags by last-write-wins, and
no decision cleans up, deletes, or archives old collections. Old collections
are retained until the owner explicitly approves any cleanup.

## Ordinary CSV consolidation has no history guarantee

<!-- latinitas-consolidation-statement begin -->
Ordinary CSV structural consolidation has no demonstrated history guarantee: a
future history-preserving migration requires separate approval,
destination-aware per-card mapping, explicit annotation and tag reconciliation,
backup and rollback, and verified native scheduling and history preservation
evidence.
<!-- latinitas-consolidation-statement end -->

Until all of that exists and is separately approved, exporting and importing a
CSV must not be claimed to preserve card histories, scheduling states, or
personal annotations of legacy notes. For notes with valuable review history
or conflicting `Personal Notes`/tags, the supported answer is to retain the old
collection and defer conversion, not to merge.

## The fresh-start checklist (rehearsed)

The only supported transition for legacy data is an explicitly approved fresh
start, and only for data the owner confirms is disposable:

1. **Back up first.** Copy the old collection (or export it) and verify the backup
   exists and is non-empty before anything else. The guard's backup check verifies
   existence and non-emptiness only; confirming the copy is current and complete
   stays with the owner.
2. **Inventory the old model.** Establish the legacy note types, note/card
   counts, personal annotations, tags, and which notes carry review history.
   Notes with valuable history or personal annotations are retained and
   deferred even under an approved fresh start.
3. **Create a new dedicated note type.** Build it from
   [reference-note-type.md](reference-note-type.md) — never reuse or modify a
   legacy exercise note type, and never point the export profile's generated
   note type at one (declare your legacy note types with
   `--legacy-note-type` on `preview`/`generate --profile`; a match is rejected
   as an incompatible legacy profile).
4. **Approve explicitly.** A fresh start requires all four confirmations: an
   existing non-empty backup file, a new dedicated note type distinct from every
   legacy note type, confirmation that the legacy data is disposable, and
   acknowledgement that **every fresh-start schedule is new** — new identities
   mean new cards with no review history.
5. **Export into the new note type.** The generated notes mint new
   `latinitas-v2-` identities; old per-exercise identities are never mapped,
   reused, or carried over.
6. **Keep the old collection.** Retain it untouched (the guard never mutates
   it and never schedules cleanup); archive or delete it only under separate
   owner approval.

The synthetic rehearsal in `tests/unit/legacy_transition_test.py`
(`test_synthetic_fresh_start_rehearsal_follows_the_documented_checklist`)
walks this checklist end to end against synthetic evidence: it copies and
verifies a synthetic collection backup, plans the transition of disposable,
history-carrying, and personally-annotated legacy notes, confirms the
disposable note alone gets the `fresh_start` decision, and asserts the old
collection bytes are unchanged afterwards.

## Regeneration is not migration

Compatible regeneration and migration are different operations:

- **Regeneration** applies only to notes already on the current model
  (`latinitas-v2-…`, `Note Schema` 3). It preserves the note's identity and
  therefore its cards and review history; the prior-export checkpoint
  withholds any note row that would lose a previously exported card key
  rather than letting an import drop the card (see
  [deterministic-csv-export.md](deterministic-csv-export.md)).
- **Migration** — restructuring legacy per-exercise notes into the current
  model while preserving their histories — does not exist in v0.1.0 and has
  no demonstrated CSV-only implementation. A future history-preserving
  migration needs the separately approved plan described above.

## Inventory of affected tests, examples, and fixtures

T-034 adds no live collection mutation; these existing assets touch legacy
models and stay in scope of the inventory:

| Asset | Legacy relevance |
| --- | --- |
| `tests/unit/cli_test.py` | The experimental `split` path that mints `legacy-split-guid-v1-` clones (retained as legacy/experimental, not part of the transition workflow) |
| `tests/unit/sources_test.py` | Legacy-schema COLPKG/SQLite source fixtures (`_write_legacy_database`) for the source adapters — reading old collections as sources, not converting them |
| `tests/unit/manifest_test.py` | Legacy unscoped schema-1 CSV identity manifests, gated behind an explicit scope approval |
| `tests/unit/preview_export_test.py` | Legacy unscoped manifest fresh start: the identity-state precedent for explicit re-confirmation |
| `tests/fixtures/extraction-review-corpus.csv`, `tests/fixtures/representative-university-latin.apkg` | Synthetic/sanitized sources; no legacy destination note model |
| `docs/reference-note-type.md` | The new dedicated note type this policy requires for fresh starts; its import warnings (tag replacement, `Personal Notes` unmapped) apply unchanged |
| `docs/principal-part-parsing.md` | The legacy `supine` profile default retained for compatibility |

## Scope and limits

- The v0.2.0 offline managed destination contract rejects historical versioned
  identities and incompatible schema/template layouts before adoption or apply.
  Errors offer retain-and-review or a backed-up, explicitly selected fresh start
  in a separate destination, never slot repurposing or manifest reinterpretation.
- Library callers use `plan_destination_fresh_start(profile, evidences,
  original=original_binding, destination=new_binding, selection="fresh_start",
  fresh_start=approval, legacy_note_types=inventory_names)`. The original binding
  is reviewed inventory (its historical schema is not adopted); the new binding
  must match the current frozen schema/template contract. Portable collection
  IDs must differ, and the new dedicated note type must not reuse the inventoried
  original type. The approval's `dedicated_note_type_id` must match the new
  binding's collection-local schema ID, explicitly associating that ID with the
  approved display name (`dedicated_note_type`, matching the profile). ID and
  display name are distinct namespaces, not compared for equality.
  Backup, disposable-data and new-schedule confirmations remain
  mandatory, including when the inventory is empty. Missing selection and
  scheduling-preserving consolidation requests fail clearly.
- This API extends the four read-only decisions above. It reports both destinations
  and explicitly discloses no inherited scheduling. It neither exports/imports
  cards nor authorizes structural CSV effects; it leaves original notes/cards,
  personal fields, tags, source structure, review logs and identity manifests
  untouched. Valuable-history and personal-annotation notes still retain/defer.
  Native isolation was checked in the T-061 run
  ([managed-csv-native-verification.md](managed-csv-native-verification.md)) for
  that fixture and client only; it is not a general certification. The planner
  is a library API with no CLI command.
- v0.1.0 ships the guard, the policy, and the rehearsal. It does not ship
  migration machinery, destination-aware merging, collection cleanup, or any
  live collection mutation.
- Broad Anki-version compatibility certification remains v0.9.0 work; the
  native multi-card verification pass is tracked as T-035.
