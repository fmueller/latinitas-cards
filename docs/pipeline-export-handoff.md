# Pipeline export receipts and manual resolution

`pipeline export` returns both a batch `receipt_id` for replay/continuation and an
`export_receipt` with format `latinitas-export-receipt`, version 1. Its nested
`receipt_id` hashes that record without its own ID. Keep the run store and the
original CSVs: later exports validate the recorded file hashes and receipts.
Batch options `--after-receipt` and `--reconcile-receipt` take the **batch** ID,
not the nested export ID. The export receipt is committed inside the same durable
publication as the CSV and batch receipt, rather than published as another file.

Each exported card lists its semantic note/card identity, decision IDs, assertion
IDs, rendered-content digest, acceptance origin and owned fields/tags. Decision
IDs bind the assertions to their evidence, source, checker versions and policy.
Legacy `grammar-v1` owns no acceptance tags; opt-in `acceptance-1` receipts own
the managed tags described below. Neither receipt installs a layout or approves
any destination.

Regeneration checks every prior export in this run, including earlier partial
batches. Ineligible cards retain their previous authority IDs and reasons in
`export_receipt.ineligible`. New CSVs withhold the whole note row if a previously
exported sibling is unresolved, leaving unrelated notes exportable. A changed
selection or processing ceiling is not proof that an installed card became
ineligible: missing current coverage is reported as unresolved. Old batch-only
receipts lack acceptance commitments and require manual inspection.

Use `pipeline handoff --run RUN [--profile PROFILE] [--json]` to inspect these
actions without another export, including when source files cannot be loaded or
confirmed. It returns `state: not_applied`; installation is presumed from export,
not observed. `pipeline report` displays the handoff assessment saved in the
orchestrated execution snapshot, not a fresh reassessment. Historical reports
without that assessment say "not assessed", rather than implying no actions.
Unresolved actions remain in later receipts. A new export is not a resolution
record, and this release does not accept an unverified "resolved" flag.
Replay preserves historical files, but requires explicit reconciliation if the
current unresolved actions differ or affect rows in the old CSV. Distinct
specific causes are retained in deterministic JSON-array text in `reason`;
missing current coverage does not erase a known invalidation cause.

## Manual destination workflow

1. Back up the collection or use a disposable collection. Separately confirm the
   grammar note type and initial import. CSV headers do not install templates.
2. Locate each affected note/card identity in the actual destination. Inspect
   the stale assertion/status and its reason; omission from CSV has not removed it.
3. Manually update/remove stale assertions and badges, or manually suspend the
   affected card if appropriate. Preserve other siblings, Personal Notes, other
   user fields, all unowned tags and their case, scheduling and review history.
4. Before **any** existing-note reimport, inspect a fresh destination and reconcile
   its full tag set, including destination-only tags and namespace collisions.
   Explicitly adopt collisions or stop; approve the exact proposed update only
   after preservation is checked. These CSVs are not safe for direct reimport.
5. Record the manual resolution and verify the actual collection afterward.
   Offline receipt tests cannot prove native application or preservation.

There is no direct Anki integration, CSV suspension/retirement or automatic
clearing/template migration. No destination write or managed plan is emitted by
this manual path. Existing managed retained-field workflows still require their
own supported layout, destination snapshot, exact approval and post-import
observation/reconciliation; an export receipt supplies none of those permissions.

The shared receipt ledger also accepts separately assessed curated approval IDs
and invalidation reasons. Exact rule/mnemonic/rendered-approval checking belongs
to the curated consumer; a generic human decision cannot approve curated content.
Synthetic ledger/automatic-policy fixtures are engineering controls, not semantic
evaluation or policy enablement. Ordinary automatic learning export remains disabled.

## Opt-in acceptance answers

`pipeline migration-plan --acceptance --family grammar --json` prints the
`acceptance-1` field/template/CSS proposal; use `--family curated` for curated
cards. Neither command installs anything. The family-specific template digest
must match the profile's layout, with exactly `id: acceptance-1`, `version: 1`
and `template_sha256`. After manual installation, explicitly attest it with
`pipeline confirm-layout --profile FILE --family grammar --installed --json`
(or `--family curated`). This writes a local `FILE.acceptance-1.json` sidecar
bound to that profile path, family and layout, without altering the approved
profile schema. It is an attestation, not observed/native installation evidence.
Missing, moved, conflicting or stale confirmation blocks the opt-in layout.
Legacy three-key layouts remain valid and unchanged. Curated and grammar
digests are different. For combined `curated` preview/export, supply the curated
`--profile` and a separately confirmed grammar `--production-profile`, alongside
`--production-run` (and `--production-output` for export). Explicit language,
preset and recipe overrides apply to both; each retains its own inputs, policy,
selection and relative-path base. Invalid production layout configuration is
checked before curated publication, but the two exports remain separate
transactions with explicit recovery reporting for later failures.

Back up the actual collection first. Append the three managed fields `Acceptance
Data`, `Evidence Summary`, `Acceptance Receipt`; preserve the first identity
field, semantic CardKey, every existing field position, template order and
user-owned fields. The complete installed schema is the old field sequence,
including Personal Notes, followed by the three new fields. Map CSV columns by
field name; leave Personal Notes unmapped. Install the
proposal only by explicit opt-in. The exporter does not write Personal Notes in
this layout. Offline identity/ordinal checks do not prove scheduling or review
history preservation; native import/update acceptance remains T-201.

Answers show each card's own human/automatic origin and a native keyboard-operable
disclosure: `Warum diese Antwort?` or `Why this answer?`. Evidence is collapsed
initially, with basis/check summaries and limitations, not private reasoning or
model-confidence percentages. Full references remain in the stored report.
Curated candidate renderings say that the approval route is human, not that an
unreviewed candidate has passed. The separately labeled `approval_back_html`
shows the proposed learning appearance if approved, including its human status
and required completed-check wording; it is not current approval. The rendered
fingerprint commits to that exact prospective answer and front. Only after all
matching human approvals does the unchanged prospective answer enter learning
CSV. Decision IDs stay in metadata rather than changing the approved appearance.

`Acceptance Data` stores version 1 per-card decision IDs, origin, basis, checks
and limitations; `Evidence Summary` stores escaped per-card HTML; `Acceptance
Receipt` identifies the local run and card keys, not the circular final receipt
digest. These fields never supply acceptance authority. Namespaced tags summarize
all currently assessed siblings: human only when all are human, automatic only
when all are automatic, mixed otherwise, incomplete when one is unresolved.
Basis tags indicate at least one sibling's basis. Source tags retain their case;
an unowned acceptance/basis namespace collision blocks export, conservatively
including case variants. Grammar source tags contribute to the export snapshot,
not linguistic acceptance: tag-only updates require explicit reconciliation
instead of silently replaying an old request. Collisions are checked before
replay/recovery. Destination-only
tags still require the manual destination workflow above before reimport.

Morphology uses the deck profile's existing terminology conventions, muted or
monochrome themes and light/dark appearance. Answers carry accepted analyses as
text; fronts receive no analysis decorations. Labels, weight and underlining do
not depend on color alone. Presentation-only theme/layout choices preserve
linguistic decisions and semantic identity; terminology/content changes still
need their evidence. A curated theme/layout change invalidates the exact rendered
approval while retaining unchanged rule approval. No syntax/construction palette
or automatic template replacement is included.
