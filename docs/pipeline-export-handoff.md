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
The grammar layout currently owns no acceptance tags; this receipt does not
install the later acceptance presentation layout or approve any destination.

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
