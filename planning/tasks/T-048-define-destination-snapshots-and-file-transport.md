---
id: T-048-define-destination-snapshots-and-file-transport
title: Define destination snapshots and file transport capabilities
status: completed
priority: high
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-05T00:09:25Z"
---

# T-048-define-destination-snapshots-and-file-transport Define destination snapshots and file transport capabilities

## Description

Record the minimum destination snapshot, applied-baseline, and operation-by-transport contracts before implementing managed updates. The agreed initial scope is compatible content/tag CSV updates; structural additions, retirement, reactivation, and consolidation are planned but unsupported for application unless separately proven safe.

## Acceptance

- Specify destination binding to collection/export, Latinitas-owned note type/schema/template registry, and managed note set; include identities, managed values, tags, completeness, fingerprints, and freshness. Identify what card/suspension/review evidence is available and which operations its absence blocks.
- Define a practical offline snapshot acquisition and post-import observation path, distinguishing note-only CSV evidence from card/history evidence. Do not read or write a running collection.
- Publish a matrix for create/content/tag updates, card addition, retirement, reactivation, and migration across plain CSV, destination-aware CSV, and APKG. Unproved matching or structural capabilities are unsupported, not inferred from copied numeric IDs.
- Define explicit first-adoption reconciliation, successful observed import, partial application, retry, and baseline advancement. Prior-export checkpoints and generated files are not applied baselines.
- Document the fresh-snapshot/no-intervening-edits requirement for GUI import, the limit on enforcing that interval, and the circumstances requiring replanning or withholding preservation claims.
- Name sanitized native import checks required for each supported capability. No live integration or general template provisioning is brought forward from v0.6.0.

## Verification

Review the contract against every managed-update and lifecycle acceptance case in v0.2.0; record unresolved evidence explicitly rather than marking capabilities supported.

## Implementation and workflow-v3 evidence

- Pinned spec: `specs/v0.2.0.md`; fetched base matched clean local main and
  origin/main before start. Taskrail validate and next JSON confirmed this task,
  no active owner, and the pinned active spec before the start transition.
- Step 1: read acceptance and managed-update, lifecycle, destination-tag, safe
  application and migration spec cases; consume authoritative identity,
  schema 3 / registry v1 and reference import guidance.
- Step 2: documentation-only contract in
  `docs/destination-snapshots-and-file-transport.md`, linked from the spec.
  No production code, test changes, or red/green TDD execution is claimed.
  Contract validation uses concrete state/evidence walkthroughs and review.
- Step 3: `uv run ruff check` passed; `uv run mypy` passed (65 source files);
  `uv run pytest -v` passed (627 tests). `git diff --check` passed.
- Step 4: dedicated Task loaded `code-simplifier`; removed a redundant
  acceptance-summary list without weakening gates. Re-read the changed prose;
  documentation checks and the full validation chain passed afterward.
- Step 5: separate parallel read-only Tasks loaded `code-reviewer` in General,
  Security and Database lanes. General loaded ECC code-reviewer/common rules;
  Security loaded ECC security-reviewer/security-review; Database loaded ECC
  database-reviewer/postgres-patterns/database-migrations. General is required;
  Security covers preservation/trust boundaries; Database covers persisted
  observed anchors and partial/retry integrity. No language/framework lanes:
  no code/framework change. No additional specialists were material.
  General and Security each concluded: "No concrete task-relevant findings."
  A fresh candidate-validation Task validated DB-1; no candidates rejected.
- Step 6, DB-1 (medium, domain), verbatim finding: "Define how confirmed
  partial operations advance persisted baseline state, while keeping
  unconfirmed operations pending and preventing whole-plan advancement."
  Evidence: former contract rows retained the whole prior baseline after
  partial import, conflicting with T-054/T-059 confirmed-operation acceptance.
  Fixed: atomic confirmed note-entry/receipt advancement, unresolved anchors
  unchanged, whole-plan status separate, mixed within-note outcomes blocked,
  interrupted persistence and backup restoration reconciled before retry.
  No deferrals. Documentation fix, not an invented executable TDD transition.
- Step 7: fresh disposition-verification Task loaded `code-reviewer`, General
  ECC lane; concluded "Disposition: RESOLVED" and "No new task-relevant
  findings." One review/fix/recheck cycle. Final full chain passed: ruff,
  mypy (65 files), pytest (627 passed); `git diff --check` passed. Inline
  documentation checks passed: 3 local links, authoritative schema/registry,
  seven matrix rows across three transports, only two conditional CSV
  capabilities, all APKG unsupported, DB-1 granularity/recovery coverage.

### Concrete acceptance walkthroughs

- Fields: B=old, D=old, P=new permits approved update; B=old, D=new,
  P=new is convergent/no-write; B=old, D=manual, P=old or new is conflict.
  Unknown baseline requires adoption, not overwrite. Wrong destination,
  duplicate/missing identity, unknown schema, stale or incomplete evidence
  blocks/reconciles; only complete absence proves a create plan, whose apply
  is unsupported initially. Unchanged regeneration is no-op, not permission.
- Partial stop: observed A old-A to new-A advances only A; B remains old-B.
  Abandoning the rest and proposing newer-A uses confirmed new-A, avoiding
  DB-1's false conflict. Mixed field/tag results remain unresolved; restored
  old-A invalidates/reconciles the new-A anchor. Generated-only, missing
  observation and interrupted persistence never imply whole-plan success.
- Tags: overlapping source/configured origins survive one-origin removal;
  manual additions survive; user-deleted required tags conflict; explicit
  keep overrides survive removal. Unknown ownership/lifecycle collisions
  require review. Exact full sets, not generated-tag inclusion, are the gate.
- Lifecycle: distinct objects/senses stay distinct; siblings retain frozen
  bindings and independent full histories. Eligibility loss/addition is not
  compatible content apply. All four reactivation cases require explicit
  provenance/approval; structural effects remain unsupported. Old layout is
  rejected; separate fresh start preserves original data and discloses new
  scheduling; consolidation is unsupported, not ordinary regeneration.
- GUI freshness cannot be locked/proven by file output. Fresh evidence plus
  no-edit interval is required; unknown interval withholds preservation
  claims. Full before/after native tables, source/unmanaged data, personal
  fields/tags and configured burying are required evidence, not claimed here.

### Remaining evidence

Sanitized native import checks are specified, not executed by this contract
task. Conditional managed content/tag support requires those checks, including
unchanged-field tag-only behavior. Structural/APKG matching, retirement,
reactivation and consolidation remain unsupported. Existing T-054/T-059 and
native-validation tasks own implementation/evidence; no new follow-up or
second task is needed. No runtime behavior or release readiness claim changed.

## Implementation Notes

- 2026-10-05T00:09:14Z: verification pass
- 2026-10-05T00:09:25Z: Workflow-v3 step 8: contract review and final gates passed before verification at 2026-10-05T00:09:14Z. DB-1 fixed and disposition independently verified. Documentation only; no native execution or TDD claimed. Compatible content/tag CSV remains conditional on native gates; structural/APKG effects unsupported.
