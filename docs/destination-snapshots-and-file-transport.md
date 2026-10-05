# Destination snapshots and file transport contract

This is the contract-first decision for [v0.2.0](../specs/v0.2.0.md#safe-update-application),
not an implemented native snapshot acquisition command or a claim of verified managed import support.
The initial application scope is **compatible content/tag CSV updates to existing
notes**. Matching, freshness and native verification gates below must pass before
advertising that capability. Unproved operations remain unsupported.

Consume the [learning-object identity](stable-generated-note-identity.md) and
[reference note type/import checklist](reference-note-type.md), not the historical
per-exercise layout. One object is one note with independent sibling cards; neither
visible text nor a deck name identifies a destination. No running collection is read
or written. No live integration or automated template provisioning is added from v0.6.0.

## Minimum snapshot evidence

The following logical record is required; a future serializer must version it. It is
not a new wire schema or a replacement for the authoritative field/slot definitions.
Missing evidence is explicitly `unknown`, never an empty set or permission to overwrite.

| Record | Required evidence and binding |
|---|---|
| Envelope | Contract version; snapshot ID; acquisition UTC time; client/version; export method/options; source artifact SHA-256; evidence level (`note-only` or `collection`); completeness and freshness status. |
| Destination | Persisted operator-assigned collection binding, profile/export provenance, and selected dedicated Latinitas note type. Collection file paths, deck labels and local numeric IDs alone are not portable collection identity. A copied collection needs explicit binding reconciliation, not automatic reuse of its baseline. |
| Schema | Actual note-type ID bound only within that collection; ordered field names/ownership, note schema version, template names/ordinals/semantic keys, registry version/digest, and actual front/back/CSS digests. Compare with `notes.AUTHORITATIVE_NOTE_FIELDS` and `cards.TEMPLATE_REGISTRY`, not the note-type name alone. Initially schema 3 / registry v1; unknown, reordered, repurposed or incompatible slots block normal apply. |
| Managed set | Explicit source scope and immutable source/object-to-`LatinitasID` membership; selection/export query and exclusions; expected versus observed note counts; evidence the whole bound set was exported, including retired notes. Do not filter only currently generated or active notes. Parser failure/incomplete input is not evidence of retirement. |
| Notes | Unique `LatinitasID`, immutable source/object provenance, destination native GUID and local note ID when available, note-type binding, exact managed field values, exact destination tag set, and modification evidence when available. Missing/duplicate identity or source/object ambiguity requires reconciliation. Absence means create only when complete membership evidence proves it. |
| Ownership | Applied-baseline reference/version, separate source-inherited and configured tag contributions, lifecycle provenance, explicit keep-as-user-owned overrides and conflict decisions. Destination namespace prefixes do not prove tag ownership. Personal Notes and other user-owned fields are excluded from managed values/writes/baseline ownership; retain local comparison evidence separately. |
| Cards (collection evidence only) | Parent GUID/local note ID, local card ID, semantic task and frozen slot binding, deck/sibling relationships, full card rows (including scheduling and suspension), full review-log rows, deck options/burying configuration, and expected card coverage. Numeric IDs are bindings inside this collection, not matching instructions for APKG. |
| Suspension provenance | For each lifecycle effect: pre-retirement suspension state, explicit approval, observed tool-applied suspension, and evidence of subsequent user changes. A currently suspended card alone cannot establish ownership. Unknown/conflicting provenance blocks unsuspension. |

Fingerprint the canonical evidence payload using SHA-256: deterministic key ordering,
notes ordered by identity, tags as sorted exact sets, fields in schema order, cards/logs
ordered by their destination IDs. Include binding, membership, schema/templates,
managed values/tags, evidence level, completeness, and available card/history/user-data
comparison digests. Record capture time and artifact digest separately; omit the
self-referential fingerprint field. Never compare a note-only fingerprint with a
collection fingerprint as equivalent evidence. An export hash proves file integrity,
not that it is complete, fresh or from the claimed destination.

Keep the recoverable collection backup and full personal/card/log evidence local;
sanitized review artifacts use synthetic identities and personal text. Personal Notes
may be compared by digest but must never become importable fields or generator-owned
baseline values. A full native table comparison is required for preservation claims;
hashes/counts of selected scheduling columns alone are insufficient.

## Practical offline acquisition and observation

1. Stop reviews, sync and edits on the target and other devices for the import interval.
   Record target binding, client/version, schema/registry and managed-set selection.
   Make a recoverable full collection export **including scheduling**, through Anki's
   supported export UI. Close Anki before inspecting any collection data. Retain an
   untouched backup; analyze only a disposable copy of the closed export, never a
   live `collection.anki2`. If export omits required history/options, mark those
   records unknown and obtain a closed full backup containing them.
2. Export Notes in Plain Text with all selected note fields and tags, without losing
   retired notes or identities. A note-only CSV/text export can supply values and tags
   for planning; it cannot attest to native GUIDs, template bindings, cards, suspension,
   schedules or history unless separate closed-export evidence supplies them. Record
   selection/options, exact field mapping, counts and evidence of completeness. A
   manually supplied metadata sidecar is a reviewed assertion, not native proof.
3. Inventory the closed full export's note types, notes, cards, review logs and deck
   options. Bind GUIDs and collection-local IDs to logical identities and registry
   slots. Reject unknown schemas, duplicate identities and incomplete coverage; do
   not guess missing history or suspension provenance. This specifies evidence needed
   by later tooling, not a promise that the current source importer extracts it all.
4. Generate an inspectable plan against this snapshot and an observed applied baseline.
   Record effective profile, proposal, field/tag diffs, per-card effects, conflicts,
   unsupported operations and exact transport/client settings. Save approval bound to
   the plan digest, snapshot fingerprint, destination and baseline version. Backup and
   recovery instructions are prerequisites, not afterthoughts.
5. Immediately before GUI import, reacquire relevant destination evidence and compare
   with the approved preconditions. Changed values, tags, membership, schema, card
   state or baseline version require replanning and new approval. Open only the bound
   collection; use the reference first-field/Note Type matching and deliberate mapping.
   Omit Personal Notes entirely from importable data, even as an empty column.
6. After import, before further reviews/sync/edits, capture the native import report,
   export again, close Anki and inventory the new closed copy. Compare actual values,
   full tags, identity/card/history tables, schema and user-data evidence with the
   approved expected delta. Preserve before/after artifacts and record unexpected or
   unobservable differences. Only this observed result can advance an applied baseline.

The GUI/file transport cannot lock the collection or enforce the interval between
snapshot and import. Operator confirmation of no intervening edits is required; there
is no arbitrary time-to-live that makes an old snapshot safe. A timestamp, import success
dialog, or equal post-import values cannot prove that an intervening edit was not
overwritten. If the fresh capture/no-edit interval cannot be established, **withhold
preservation claims and replan**, rather than mark the apply successful. Suspected
sync, review, tag edit, wrong profile or stale file invalidates approval. This limitation
must appear in the application report even for an otherwise observed successful import.

## Operation-by-transport matrix

`Conditional` means the intended initial supported scope, **not yet evidenced managed
support**: all gates and native checks in the next section are required. `Legacy` means
the limited existing source-only workflow, with no destination-preservation guarantee.
`Unsupported` means refuse that managed apply effect, even if Anki can technically
import the file. An approved compatible content-only subplan may proceed separately.

| Operation | Plain source-only CSV | Destination-aware CSV | APKG |
|---|---|---|---|
| Create new object/note | Legacy fresh import in a separately chosen destination; new scheduling, no managed adoption | Unsupported initially; plan may show create, not authorize it | Unsupported managed application; GUID/model matching unproved |
| Existing compatible content update | Legacy reimport; tag-destructive, not safe managed apply | Conditional: unique first-field identity + Note Type match, same schema/slots and eligible card set; three-way field approval | Unsupported; GUID, note-type/template and modification-time semantics unproved |
| Existing note tag update | Legacy replacement/unchanged-row behavior; no manual-tag preservation | Conditional: reconciled full tag set and proven tag-only behavior; no lifecycle suspension claim | Unsupported; tag merge/update behavior unproved |
| Add eligible sibling card | Unsupported | Unsupported initially, even when changing a guard can cause native creation | Unsupported until actual destination identities/histories and only expected added rows are proven |
| Retire one card or entire object | Unsupported | Unsupported: cannot suspend; a tag or cleared required field is not retirement | Unsupported until suspension and surviving histories are proven |
| Reactivate existing retired card | Unsupported | Unsupported: cannot reverse suspension | Unsupported until approved tool-owned-only unsuspension and identity/history retention are proven |
| Schema/template migration or old-note consolidation | Unsupported | Unsupported | Unsupported until complete mapping, user-data/history reconciliation, rollback and retry are proven |

Content-only updates must not change conditional eligibility or introduce/remove cards.
Preserve last safe content/provenance when ineligibility would require retirement; label
retained claims as stale/pending, not newly approved. If compatible schema cannot express
safe retention, block apply. Changing `Enabled` or required-role values is not a loophole
for unsupported lifecycle operations. Lifecycle-tag collisions require review; setting
`latinitas::retired` alone must never imply approved or completed suspension.

No scheduling guarantee follows from copying numeric note/card/model IDs into an APKG.
Future APKG proof must establish native GUID matching, compatible note type and template
bindings, update/modification-time behavior, and preservation of actual destination rows.
Experimental APKG commands are not evidence of managed application support.

## Adoption, application and retry state

An applied baseline records **observed accepted managed state**, not the last generated
proposal. It is versioned and bound to the destination/schema/managed set. Store managed
values, source/configured tag contributions and keep overrides, immutable identities,
observed snapshot/result references, approvals/conflict decisions and operation receipts.
Card effects require separate observed lifecycle/provenance evidence. Never store
Personal Notes as owned baseline values. The v0.1 prior-export eligibility checkpoint
is not an applied baseline and cannot be promoted just because a CSV/manifest exists.

The initial reconciliation unit is one approved note update: its exact managed field
delta and full reconciled tag set, together with unchanged identity/card/history/user
data checks. Mixed outcomes within a note (fields applied, tags skipped, for example)
leave that operation unresolved; they do not advance its anchor without a separately
reviewed reconciliation. Independent confirmed note operations may advance their
entries in a new baseline version atomically with their receipts/evidence references,
while unresolved entries retain their previous anchors. Whole-plan completion is a
separate status and remains partial/pending until all operations reconcile. Thus partial
application never promotes all proposed values as though the whole import succeeded.

| State/case | Required decision and next state |
|---|---|
| No baseline / first adoption | Inventory complete destination evidence; reconcile source/object identities and schemas explicitly, never by visible text. Review every managed value and tag origin/keep decision. Accept observed destination values as the initial anchor only with explicit adoption approval and recorded evidence. Unknown ownership/conflicts remain blocked. Empty destination may be established by complete evidence, but initial create apply is unsupported. |
| Plan | Compare baseline B, destination D and proposal P per managed field: D = B allows approved change; D = P is convergent/no write; D differs from both B and P is conflict (including P = B destination-only edits). Resolve by explicit keep destination, accept proposal or reviewed replacement. User fields are never writable conflict choices. |
| Tags | Reconcile source/configured contributions independently, preserving destination-only tags. Removing one origin must not remove another's contribution. A user-deleted still-required managed tag is conflict; a user intending to retain an already managed tag is unobservable, so expose removal and allow keep-as-user-owned. Unknown first-adoption ownership and reserved lifecycle collisions block unresolved changes. Compare exact final sets. |
| Approved / generated | Bind approved operations to the snapshot/plan/baseline; emit only authorized compatible managed fields and full reconciled tags. Required unchanged import columns use resolved destination values, not unapproved regenerated values. File creation is pending application, not success or a baseline update. |
| Observed full success | Check every approved effect, unchanged identities/cards/logs/user fields and preserved tags. Advance the whole baseline atomically only after complete matching post-import evidence and valid freshness interval. A no-op may record a receipt without changing managed state. |
| Failed, partial or uncertain import | Record attempted operations, native report, observed outcomes and unresolved rows. Advance only confirmed reconciled note entries and provenance; keep unresolved anchors unchanged and whole-plan status partial. Missing post-import evidence means unknown, not failed-as-empty or successful. |
| Retry or abandon/replan | Reacquire a complete fresh snapshot and reconcile the current per-note anchors, P, receipts and actual D before a new plan/approval. Already applied exact results are no-write; unapplied values still at B may be proposed again; divergent values or ambiguous partial tag changes require review. Confirmed anchors remain valid if the rest of the plan is abandoned. Do not blindly replay the CSV or treat partial completion as whole-plan success. |
| Recovery | Import and baseline persistence are not one transaction. If import succeeds but observation/receipt/baseline persistence fails or is interrupted, reacquire evidence and reconcile before recording success or retrying; a CSV rollback is not destination rollback. Restore the recoverable backup only by explicit operator decision after checking intervening work. Observe restored state and reconcile or invalidate inconsistent anchors/receipts before replanning; never infer rollback from a failed dialog or silently discard later user changes. |

Concrete partial-stop check: baseline note A has `Meaning=old-A`, note B has
`Meaning=old-B`; approved proposals are `new-A` and `new-B`. Observe only A's complete
operation (including its full tags and preservation checks). Persist A's anchor as
`new-A`, retain B's `old-B`, and report the plan partial. Abandon the remaining plan.
With fresh D(A) = `new-A`, a later approved proposal `newer-A` is a safe update from
the confirmed anchor, not a false divergent conflict against `old-A`. If A's tags were
skipped, its operation is instead unresolved. If a backup restores A to `old-A`, the
`new-A` anchor is inconsistent and must be reconciled or invalidated before retry.

Old per-exercise layouts are rejected by normal apply. An explicitly chosen fresh start
in a separate destination leaves original notes/cards/user data untouched, discloses
that new cards inherit no scheduling, and is not adoption or migration. Consolidation
requires a separate approved old-note/card/task mapping, full Personal Notes/provenance
reconciliation, union of user tags and preserved histories; ambiguous senses, duplicate
task histories or incompatible templates block it. It remains unsupported here.

## Required sanitized native checks

Run these on a disposable synthetic collection, not a user's running collection.
Record tested transport, Anki client/version, OS, import options/mapping, registry/schema,
before/after closed exports, import report, exact table diffs and limitations. The existing
v0.1 native results are regression context, not proof of v0.2 destination-tag or lifecycle
safety. No new native execution is claimed by this contract task.

T-061's [native backend verification](managed-csv-native-verification.md) records
the later Anki 26.9.3 API run, exact fixture/settings, observed capabilities and
remaining limitations. It does not certify the Desktop dialog or AnkiMobile.

| Capability/gate | Checks required before declaring support |
|---|---|
| Common CSV matching and protection | Use two objects with identical visible lemmas but distinct IDs, multiple siblings with unequal schedules/review histories, a user-suspended card, synthetic Personal Notes and destination-only tags. Prove first-field + Note Type matching updates only the intended existing note, creates no duplicate notes/cards, leaves source/unmanaged notes and schema/templates intact, and never maps Personal Notes. Reject wrong collection, missing/duplicate identities and incompatible old layout before import. |
| Compatible content CSV | Exercise safe update, convergent/no-write, divergent and destination-only edits, reviewed resolutions, unchanged regeneration and no-op reapplication. Compare full card/review-log rows exactly for surviving siblings; compare all note columns with only enumerated approved fields/tags and documented native modification metadata allowed to change. For a true no-op, compare full note/card/review-log tables and explain any native churn before claiming no-op safety. Verify sibling relationships, independent schedules and configured burying without simulating the scheduler. Reject any eligibility-changing proposal. |
| Destination-aware tags CSV | Test source addition/removal, configured changes, overlapping origins, manual destination additions, user-deleted generated tags, explicit keep overrides, unknown ownership and lifecycle collision. Compare exact full tag sets on the shared note. Test tag-only changes with otherwise unchanged managed fields: the legacy importer can skip them. Prove selected native settings actually apply the authorized set without fabricating a content change to force an update; otherwise mark tag-only application unsupported for that client/settings. |
| Freshness, observation, retry and recovery | Inject a stale tag/content/card/schema change before approval/import and reject/replan. Test generated-but-not-imported output, full success, partial application then stop/replan (the A/B check above), mixed field/tag outcomes within one note, import success followed by interrupted persistence, missing observation, intervening edits, no-op retry and backup restoration after confirmed success. Verify only confirmed entries advance, unresolved anchors remain unchanged, whole-plan status is not falsely complete, inconsistent anchors are reconciled/invalidated, and new approval follows reconciliation. Record that edits in the unobservable GUI interval cannot be automatically detected. |
| Any future structural capability | Separate gate, not initial support: enumerate expected card/column deltas for addition, retirement and reactivation; retain existing parent/card/slot IDs and full surviving schedules/logs. Test single-card versus whole-object retirement, user-suspended-before-retirement, unknown ownership and re-enablement without approval; only explicit tool-owned suspension reversal may unsuspend, with unchanged siblings and no replacement cards. Migration also needs complete old-to-new mapping, duplicate-history rejection, rollback and retry. |

### Open evidence

The native backend report establishes the tested content/tag and isolated tag-only
results for its exact client/settings/fixture. Other chosen import paths/settings,
including Desktop GUI import, still need their own sanitized checks before support claims.
Card addition, retirement/reactivation, APKG matching and consolidation are unproved and
remain unsupported. Later tasks implement snapshot/plan/apply contracts and obtain that
evidence; this decision does not authorize another task or widen release compatibility.

## Offline state API (T-054)

`latinitas_cards.destination_state` implements version-1 JSON evidence and a separate
`latinitas-observed-baseline` journal. It does not acquire native evidence, import files,
read a running collection, or prove transport compatibility. The synthetic test fixtures
are reviewed assertions, not Anki preservation evidence.

- `BoundDestination` selects a portable collection binding, profile, actual note-type
  schema/template digests and immutable managed-set membership. `schema_contract` supplies
  the expected reference layout, **not** evidence that a destination uses it.
- `read_snapshot(payload, binding)` validates whole-set completeness/freshness assertions,
  identities, schema, exact fields/tags and available full card/history evidence. Missing
  card/history/personal evidence is unknown; note-only snapshots can inform adoption but
  cannot confirm application. Version 1 requires no export exclusions: an identity or
  arbitrary query excluded from the bound set cannot prove absence. Direct snapshot
  construction enforces the same evidence validation. The capture fingerprint excludes
  acquisition time, snapshot ID and artifact hash; those are retained separately in
  evidence references.
- `adopt(snapshot, ownership, approval)` requires a reviewed decision for every observed
  note and tag. Ownership names source/configured contributions, explicit keep tags and
  keep fields. Personal Notes are never managed fields; only their comparison digest is
  retained. Neither prior-export manifests nor file existence establish an anchor.
- `begin_observation` records an approved existing-note target and its preconditions as
  pending. Persist it with `save_state` **before** any later external import. This is not
  permission to import: native/client, freshness, plan and transport gates remain later
  work. Unknown preservation evidence and pending effects block another attempt.
- `observe` records actual per-note results and requires explicit no-intervening-edit
  interval confirmation. Exact full fields/tags plus unchanged identity, personal data,
  cards, history and deck options confirm one indivisible operation. Independent confirmed
  entries advance in a new version; mixed/unknown outcomes keep their previous anchors.
  Whole-plan status remains partial until every operation confirms.
- `reconcile` returns anchors inconsistent with fresh evidence, including backup restore.
  Changed or unknown collection deck options also require explicit review.
  `review_reconciliation` explicitly accepts observed values/ownership, preserves historical
  receipts and abandons remaining effects. A new plan requires new approval; no file is
  blindly replayed. This also recovers imports whose result persistence was interrupted.

`save_state` validates then fsyncs and atomically replaces one journal containing both
anchors and receipts. It requires an existing parent directory and a single operator/writer;
it is not a concurrent database. On any persistence error, reload the journal and reacquire
destination evidence before deciding what happened: an error after replacement may mean the
new file exists. `load_state` rejects missing/corrupt/foreign-format data. Preserve the old
journal and recoverable native backup locally; do not overwrite a journal with a prior export.

## Offline reconciliation API (T-055)

`destination_reconciliation.reconcile_note(snapshot, state, identity, proposal,
contributions, decisions)` compares one shared note, so every sibling card sees the
same resulting tag set. It uses the existing binding and observed-anchor validators;
missing anchors and pending effects block normal reconciliation. It neither advances
the anchor nor approves application. The managed handoff may retain reviewed current
field/tag differences after exact plan/approval verification; changed preservation
evidence or inconsistent recorded successes still require `review_reconciliation`.

The result exposes exact managed `fields`, changed-field `writes`, exact sorted `tags`,
`tag_write`, proposed `removals`, `ownership`, `conflicts`, and reviewed `decisions`.
Only conflict-free results include a journal-compatible `target`. Personal Notes and
custom user fields are excluded even if supplied in the proposal; decisions targeting
them are rejected. `reconcile_values` is the pure comparison primitive for reviewed
values; use the bound-note entry point for snapshots.

Decision keys are `field:<name>` or `tag:<tag>`. Each decision records a non-empty
`approval` reference and an action: `keep_destination`, `accept_proposal`, or
`replacement` (text `value` for fields, boolean presence `value` for tags). Tags also
support `keep_as_user_owned` for a present destination tag. The result records each
decision and its exact `result`. Removal candidates remain visible even when a review
keeps them. No prefix implies ownership and no decision grants suspension permission.

The journal retains separate `source_tags`, `configured_tags`, optional `lifecycle_tags`,
and persistent `keep_tags`/`keep_fields`. Optional `suppressed_tags` records explicit
retention of a user deletion while the source/configured contribution remains required;
it must be a subset of managed contributions and absent from the final tag set. If a
user later re-adds that tag, the observable addition is preserved as user owned.
Reviewed decisions survive pending targets and successful observed anchors. Existing
version-1 journals without the optional keys remain valid. These are offline assertions,
not evidence of native tag-only import behavior, scheduling safety, or calibration.

### Offline card lifecycle planning

`card_lifecycle.plan_card_lifecycle(snapshot, state, identity, proposal, ...)` consumes
the same observed anchors and journal, not an export checkpoint. The proposal is an
existing `GeneratedNote` with the complete frozen rendered registry. Its immutable
source/object identity must match; visible lemmas never reconcile objects. Membership
may explicitly use `[LatinitasID, scope, source_id, object_key]` and note `source` uses
`[scope, source_id, object_key]`, allowing separate senses in one source entry. These
bindings are checked against the existing identity derivation; legacy three-element
members remain readable but cannot stand in for an ambiguous object split.

Complete destination card rows must include `id`, `semantic_key`, `ordinal`,
`template_name`, and (for lifecycle planning) boolean `suspended`, alongside full native
row evidence. Template bindings are checked against the frozen registry. Eligibility
is not evidence of actual card existence: even unchanged enabled fields with a missing
destination row imply an unsupported addition. The content-only journal also enforces
this gate, rejects guard-based creation/front clearing, and rejects changes to the
reserved retired tag rather than pretending tags implement suspension. Existing slot
write restrictions remain in force; this does not broaden transport capability.

Plans expose independent retain/add/retire/reactivate effects, existing card/template
bindings, explicit approvals, blockers and unsupported effects. Missing/ambiguous/
withheld/inapplicable/parser-failure proposals retain existing cards pending review.
Approved eligibility loss requires a per-key approval; whole-object retirement has a
separate approval and proposed `latinitas::retired` contribution. Single-card retirement
never proposes that note tag. All structural effects have no CSV target. Unrepresentable
last-safe slot content blocks retention; retained fields and observed provenance are an
explicitly stale/pending, not newly approved, whole-note bundle. The bundle is not mixed
into fresh shared fields while suspension is pending.

Observed retired card rows may carry `retirement` evidence with `approval`, boolean
`pre_suspended`, boolean/unknown `tool_suspended`, and `observed` snapshot reference.
These are reviewed observed facts in the existing anchor, not a second lifecycle store;
this planner never creates a receipt or advances state. Eligible reactivation requires
explicit per-key approval and the existing row. Only known, non-conflicting tool-owned
suspension of a previously unsuspended card can propose `unsuspend`; pre-suspended user
cards remain suspended and unknown ownership blocks reversal. Profile re-enablement
alone gives no authority. No plan result proves actual scheduling/history preservation:
native structural transport checks remain a separate later gate.

## Managed JSON review and approval

`managed plan request.json` writes deterministic JSON to stdout, without changing
the destination or baseline. The request contains `binding` (the selected
`BoundDestination` payload), `snapshot` (validated version-1 evidence), `baseline`
(observed journal or null), `effective_profile` (the full resolved profile including
presentation settings), and `proposals` (an explicitly selected array). Each proposal
has `identity`, complete `fields` containing only authoritative managed fields,
optional `contributions` keyed by source/configured/lifecycle tag origins, optional
`decisions` using the reconciliation API above, `status` (default `approved` means
generation/knowledge status only), and `retire` (default false). Omission never retires
an object. Membership must include its immutable object key. Generated fields can be
obtained from `GeneratedNote.to_anki_fields()` excluding Tags and Personal Notes.

Plans expose create/update/unchanged/conflict/retire classifications, reasons,
baseline/destination/proposal/resolved field differences, full tags, card effects,
blocked operations and transport capability flags. Missing baselines require adoption;
wrong, incomplete, stale, duplicate and incompatible evidence fails closed. An old CSS
digest requires separately reviewed manual setup, never automatic template migration.

```bash
uv run latinitas-cards managed plan request.json > plan.json
uv run latinitas-cards managed approve plan.json \
  --operation 'latinitas-v2-…/field/Meaning' \
  --review 'explicit apply review of this selected operation' > approval.json
```

Use the exact operation IDs from the plan; repeat `--operation` for a subset. There
is no implicit approve-all. `--review` is required and cannot be replaced by profile
confirmation, linguistic claim approval, or a lifecycle tag. A receipt binds the whole
resolved plan (including presentation), selected operations, and exact `targets`,
`import_columns` and `import_rows`. These rows are an **inspectable proposed import
footprint**, not a generated CSV or completed application. Required unselected fields
and unselected tags use destination values. Unselected notes have no row. Personal
Notes is absent from both targets and import mapping. Empty/no-op selections have no
targets or rows and do not advance a baseline.

The conservative CSV scope allows only reconciled Lemma, Principal Parts, Meaning,
Generator, Profile, and a reconciled full tag set. Slot prompt/answer/guard changes
(including changed rendered presentation), creation, retirement, reactivation and
migration remain inspectable but unsupported. Compatible content-only subsets retain
actual destination slot content; unrepresentable or uncertain retention blocks them.
`verify_approval(plan, receipt, fresh_snapshot, baseline)` checks exact snapshot and
baseline binding, recomputes derived operations, rejects changed receipts/payloads,
and checks actual destination card-set guards before a future transport can proceed.
Any changed plan, resolution, profile or presentation requires renewed approval.

The receipt is a local reviewed assertion, not a cryptographic signature or native
preservation proof. Source-only v0.1 exports remain unchanged and do not gain managed
preservation claims.

## Managed CSV handoff and observation

These commands implement the offline handoff, **not native import certification**.
T-061 remains the native safety gate. Use one operator/writer and keep the journal,
backup and before/after evidence local. No command opens a running collection.

1. Acquire a fresh complete bound snapshot as above and a recoverable full collection
   backup including scheduling. Verify the backup can be restored; a readable file
   or recorded hash alone does not establish native recoverability. Record its path
   and explicit recovery procedure. Freeze edits/reviews/sync on every device through
   GUI import and the result capture. The CLI cannot lock or prove this interval.
2. Plan and approve exact operation IDs. Include `generated_note_type` and
   `target_deck` in the approved `effective_profile`; manually verify that the name
   selects the native note type whose ID/schema is bound in the snapshot. Names alone
   do not establish matching. Any changed snapshot, baseline, schema, profile, plan,
   mapping or selected-operation footprint requires replanning and renewed approval.
3. Prepare `handoff.json` with `binding`, fresh `snapshot`, saved `plan`, saved
   `approval`, `backup` (local path), `recovery` (verified restoration procedure),
   `note_type` and `deck` (exact approved profile values). Run:

   ```bash
   uv run latinitas-cards managed emit handoff.json --state baseline.json --output updates.csv
   ```

   Emission revalidates the derived plan and approval, records pending effects and
   backup/hash before publishing a new CSV, then records emission outcome. It refuses
   existing output files and baseline/backup aliases. Required unselected columns use
   destination values; only selected notes appear. Personal Notes/user fields never
   appear. `approved`, `emitted`, `pending`, `observed`, `failed`, and `unresolved`
   lists distinguish authorization, transport output and actual result. An emitted
   file is still pending, not successful application or a scheduling guarantee.
   A post-publication persistence error returns emission `unknown` and exits nonzero:
   the CSV may already exist. Reload the journal and inspect the artifact/hash and
   destination before recovery; never treat this as permission to emit/import again.
4. Open only the bound collection. Use CSV Update with **first-field LatinitasID +
   Note Type** matching and the emitted column mapping. Map Tags to the indicated
   tags column and leave Personal Notes unmapped. Inspect the preview and supported
   client/settings; do not import unsupported structural/slot effects. Capture the
   native report, then export/close and acquire the result snapshot before any edits.
5. Prepare `observation.json` with `binding`, `snapshot` (observed result, or `null`
   when no result evidence exists), `plan_id`, and `report` (native outcome/evidence
   reference). Run:

   ```bash
   uv run latinitas-cards managed observe observation.json --state baseline.json --interval-confirmed
   ```

   Supply the flag only when the fresh-snapshot/no-edit interval is established.
   Without it even equal final values stay unresolved and no preservation claim or
   anchor advancement is made. Exact fields/tags and unchanged full preservation
   evidence confirm an indivisible note; independent confirmed notes advance
   atomically with receipts. Mixed field/tag results do not advance that note.
   A tag-only row skipped by Anki stays unresolved/pending or is unsupported for
   those native settings. Never alter unrelated fields or metadata to force Update.
6. Before **any retry**, inspect/reacquire actual destination state. Re-run observation
   to recover an import that succeeded before result persistence was interrupted;
   never blindly emit/import again. Persistence errors may occur after replacement:
   reload the journal and inspect the CSV hash and native destination, not the dialog
   alone. A CSV transaction rollback is not destination/baseline recovery.
   Interrupted emission status stays pending until its artifact and actual destination
   are checked; existing output is never overwritten.
7. For partial/mixed outcomes, an unknown interval, or restoration after recorded
   successes, acquire a fresh snapshot and explicitly review ownership for **every**
   observed note. Use `managed reconcile reconciliation.json --state baseline.json`,
   where the JSON contains `binding`, `snapshot`, `ownership` (the adoption format),
   and `review`. This accepts current observed anchors, preserves historical receipts
   as past evidence, and abandons pending effects. Replan and obtain new selected
   approval for remaining writes; confirmed results become no-write. Restore a backup
   only by operator decision after checking intervening work, then reconcile the
   restored evidence before retry. Recorded success inconsistent with restoration
   is rejected, never silently replayed.

Content/tag output remains conditional on native matching/tag-only checks. Card
addition/retirement/reactivation, schema migration, slot prompt/answer/guard edits,
and APKG application remain unsupported. Incompatible CSS requires separate manual
setup and new evidence, not automatic migration.
