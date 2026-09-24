# Stable generated-note identity

LatinitasCards keeps the logical identity of a generated exercise separate from
the text shown to a learner. The identity is the SHA-256 digest of this
canonical JSON object, prefixed with the versioned `latinitas-v1-` marker:

```json
{"exercise_key":"perfect_1s","recipe_identity":"principal_part_completion","source_identity":"source-17"}
```

The digest input is domain-separated with `latinitas-logical-id-v1`. Therefore
the identity depends only on the immutable source identity, recipe identity, and
semantic exercise key. Prompt and answer wording, glosses, HTML, tags,
generation metadata, software versions, and local Anki note/card IDs are not
inputs. A different recipe or semantic exercise key produces a different
`LatinitasID`.

Use `derive_latinitas_id(source_identity, recipe_identity, exercise_key)` from
`latinitas_cards.identity` rather than hashing rendered content.

## Source identities

- APKG and COLPKG records use the native Anki note GUID. Numeric note and card
  IDs are transport-local provenance only.
- A CSV profile with a stable source-ID column must resolve that column through
  the configured `source_id_field` strategy.
- An ID-less CSV uses a JSON sidecar manifest. Its exact, unique row
  fingerprints may be reused automatically after reordering. Edited rows,
  exact duplicates, insertions, removals, and stale manifest snapshots are
  returned as review items; they never silently receive or transfer an ID.
  The manifest stores an integrity digest for the identity-to-fingerprint
  mapping as well as the row snapshot, so a changed sidecar cannot silently
  authorize a reuse.

Call `reconcile_csv_manifest` with `approved_reuse={row_index: source_id}` or
`approved_allocations={row_index: None}` (or an explicit new ID) to record the
decision. Use `approved_removals={source_id}` for an explicit decision to retire
a missing identity. `complete` is false while any review item, including stale
manifest state, remains. Save the returned `CsvIdentityManifest` only after the
review items are acceptable; unresolved removals remain active and are shown
again on the next reconciliation. Approved removals become inactive manifest
tombstones and are not automatically allocated again. Sidecar writes replace
the destination atomically.

## Generated-note ownership

`GeneratedNote` separates:

1. immutable `latinitas_id`;
2. managed prompt, answer, and tags (`ManagedNoteContent`);
3. source relationship and location (`GeneratedNoteProvenance`);
4. recipe identity, semantic exercise key, and descriptive recipe version
   (`RecipeMetadata`); and
5. user-owned `personal_notes`.

`GeneratedNote.to_anki_fields()` emits `LatinitasID` as the first field, then
managed content, provenance, recipe metadata, and personal notes. Configure the
first field as Anki's unique matching field on the first import and enable
updating existing notes on subsequent imports. Managed regeneration must leave
the personal-notes field untouched.

Future APKG output derives its note GUID from the same logical identity with
`derive_anki_guid(source_identity, recipe_identity, exercise_key)`. The current
experimental split-card path follows that rule as well; newly allocated local
note or card IDs never participate in the GUID seed.
