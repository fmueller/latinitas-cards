# Stable generated-note identity

LatinitasCards keeps the logical identity of a generated learning object separate
from the text shown to a learner. One coherent learning object — usually one
Latin lexeme plus a relevant sense — is one note with zero or more eligible
cards. The note identity is the SHA-256 digest of this canonical JSON object,
prefixed with the versioned `latinitas-v2-` marker:

```json
{"object_key":"lexeme-1","source_identity":"source-17","source_scope":"scope-…"}
```

The `source_scope` member is present only for CSV sources (see below). The
digest input is domain-separated with `latinitas-note-family-v1`. Therefore the
note identity depends only on the immutable source identity, the persisted CSV
source scope where applicable, and the reviewed learning-object key. Recipe
selection, card semantic keys, prompt and answer wording, glosses, HTML, tags,
profiles, generation metadata, software/schema versions, and local Anki
note/card IDs are not inputs. A corrected gloss, changed wording, a different
selected recipe set, or a profile change enriches the same note instead of
creating a duplicate.

Use `derive_latinitas_id(source_identity, object_key, source_scope=…)` from
`latinitas_cards.identity` rather than hashing rendered content. Versioned,
independently specified expected-ID vectors are locked in
`tests/unit/identity_vectors_test.py`; they detect derivation drift and must
never be regenerated from the production helper.

## Card semantic keys are separate

Cards on one note keep their own stable semantic key,
`derive_card_semantic_key(recipe_identity, semantic_role)`, for example
`principal_part_completion:perfect_1s`. Card keys name the recipe plus confirmed
role that a template slot renders; they never participate in note identity.
Distinct sources never share note identity even when their visible words are
identical, and different cards of one object always share it.

## Source identities and the CSV source scope

- APKG and COLPKG records use the native Anki note GUID, which is globally
  scoped already. Numeric note and card IDs are transport-local provenance only.
- Every CSV source — whether it uses a stable source-ID column
  (`source_id_field`) or an ID-less sidecar manifest (`manifest`) — carries a
  persisted, unique, content-independent **source scope** in its identity-state
  sidecar (`<source>.latinitas.json`, schema 2). Independent sources therefore
  remain distinct even when they allocate the same local ID
  (`csv-source-000001`) or contain identical rows; filenames, file contents, and
  profiles never provide scope.
- An ID-less CSV manifest stores exact row fingerprints. Unique exact
  fingerprints may be reused automatically after reordering. Edited rows, exact
  duplicates, insertions, removals, and stale manifest snapshots are returned as
  review items; they never silently receive or transfer an ID. Approved
  removals become inactive tombstones.
- Moving, reordering, saving, or copying the sidecar keeps the same scope and
  IDs: a copied sidecar represents the same logical source. An independent
  source requires an explicit new scope.

### Source-scope bootstrap

An uninitialized, read-only preview reports that source-scope confirmation is
required; it does not mint apparently stable IDs or write state. The first
explicitly approved export (`generate --approve-scope`) allocates a unique
random scope and commits it with the source assignments and the CSV output as
one recoverable pair. Later exports reuse the persisted scope, including for
explicit CSV-ID columns.

Missing state is not permission to replace an established scope: exporting
again requires the same explicit `--approve-scope` fresh-start confirmation, and
legacy unscoped (schema 1) manifests are reported for an explicit
fresh-start/migration decision rather than silently reinterpreted. The
migration preserves fingerprint-based row assignments under a newly allocated
scope; the note IDs change because they now include the scope, which is the
documented pre-release identity model change.

## Generated-note ownership

The authoritative note schema (`latinitas_cards.notes.AUTHORITATIVE_NOTE_FIELDS`)
declares every field with its ownership category:

1. regular managed fields: `LatinitasID`, `Lemma`, `Principal Parts`,
   `Meaning`, source provenance (`Source ID`, `Source Scope`, `Source Kind`,
   `Source Location`, `Source Path`), and generation metadata (`Note Schema`,
   `Generator`, `Profile`);
2. user-owned `Personal Notes`, excluded from CSV export; and
3. special transport metadata: `Tags`, exported through the derived
   `#tags column:` directive rather than as a regular field.

The Anki note type and the CSV export order are derived from that schema;
`LatinitasID` stays first so Anki can match notes on repeated text imports, and
no generated column can overwrite the personal-notes field. Managed
regeneration keeps identities and leaves personal notes untouched.

Future APKG output derives its note GUID from the same logical identity with
`derive_anki_guid(source_identity, object_key, source_scope=…)`. The legacy
experimental split-card path keeps its own domain-separated transport GUID and
does not mint learning-object identities.
