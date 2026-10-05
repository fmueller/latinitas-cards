# Destination snapshots and file transport contract

This is the contract-first decision for [v0.2.0](../specs/v0.2.0.md#safe-update-application),
not an implemented snapshot command or a claim of verified managed import support.
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

| Capability/gate | Checks required before declaring support |
|---|---|
| Common CSV matching and protection | Use two objects with identical visible lemmas but distinct IDs, multiple siblings with unequal schedules/review histories, a user-suspended card, synthetic Personal Notes and destination-only tags. Prove first-field + Note Type matching updates only the intended existing note, creates no duplicate notes/cards, leaves source/unmanaged notes and schema/templates intact, and never maps Personal Notes. Reject wrong collection, missing/duplicate identities and incompatible old layout before import. |
| Compatible content CSV | Exercise safe update, convergent/no-write, divergent and destination-only edits, reviewed resolutions, unchanged regeneration and no-op reapplication. Compare full card/review-log rows exactly for surviving siblings; compare all note columns with only enumerated approved fields/tags and documented native modification metadata allowed to change. For a true no-op, compare full note/card/review-log tables and explain any native churn before claiming no-op safety. Verify sibling relationships, independent schedules and configured burying without simulating the scheduler. Reject any eligibility-changing proposal. |
| Destination-aware tags CSV | Test source addition/removal, configured changes, overlapping origins, manual destination additions, user-deleted generated tags, explicit keep overrides, unknown ownership and lifecycle collision. Compare exact full tag sets on the shared note. Test tag-only changes with otherwise unchanged managed fields: the legacy importer can skip them. Prove selected native settings actually apply the authorized set without fabricating a content change to force an update; otherwise mark tag-only application unsupported for that client/settings. |
| Freshness, observation, retry and recovery | Inject a stale tag/content/card/schema change before approval/import and reject/replan. Test generated-but-not-imported output, full success, partial application then stop/replan (the A/B check above), mixed field/tag outcomes within one note, import success followed by interrupted persistence, missing observation, intervening edits, no-op retry and backup restoration after confirmed success. Verify only confirmed entries advance, unresolved anchors remain unchanged, whole-plan status is not falsely complete, inconsistent anchors are reconciled/invalidated, and new approval follows reconciliation. Record that edits in the unobservable GUI interval cannot be automatically detected. |
| Any future structural capability | Separate gate, not initial support: enumerate expected card/column deltas for addition, retirement and reactivation; retain existing parent/card/slot IDs and full surviving schedules/logs. Test single-card versus whole-object retirement, user-suspended-before-retirement, unknown ownership and re-enablement without approval; only explicit tool-owned suspension reversal may unsuspend, with unchanged siblings and no replacement cards. Migration also needs complete old-to-new mapping, duplicate-history rejection, rollback and retry. |

### Open evidence

Managed content/tag application, especially unchanged-field tag-only
import semantics, still needs the sanitized native checks on the chosen client/version.
Card addition, retirement/reactivation, APKG matching and consolidation are unproved and
remain unsupported. Later tasks implement snapshot/plan/apply contracts and obtain that
evidence; this decision does not authorize another task or widen release compatibility.
