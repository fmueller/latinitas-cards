---
id: T-184-design-v0-5-0-acceptance-and-pipeline-contracts
title: Design v0.5.0 acceptance and pipeline contracts
status: completed
priority: high
spec_ref: specs/v0.5.0.md#open-technical-design-checkpoints
dependencies: []
updated_at: "2026-11-12T21:53:03Z"
---

# T-184-design-v0-5-0-acceptance-and-pipeline-contracts Design v0.5.0 acceptance and pipeline contracts

## Description

P50-1 contract-design draft, not implementation or policy approval. Inventory
claim_review/claim_sidecar, phrase_inputs/phrase_recipes, authored_notes, profiles,
corpus facts, CSV and managed interfaces. Own the shared contract proposal before
consumers. Follow the spec's execution-isolation and final-validation rules:
separate planning approval and active v0.5.0 are required before execution.

## Acceptance

- Propose exact command/profile names, stage/report/decision/receipt schemas,
  strict-format adapters/upgrades, compatibility and unsupported-version behavior.
  Do not overload reviewer or annotation-write auto.
- Define source/extraction/task/language/configuration bindings and producer,
  analyzer, prompt, checker and policy versions; invalidation/reuse, full-assertion
  coverage, human versus automatic states and conservative mixed-note tags.
- Define deterministic supported-pattern selection/suppression, full eligible
  export without an implicit output quota, processing/output limit precedence,
  optional batch continuation and reuse without repeated valid model checks.
- Specify separate proposal/check endpoints, hosted permission/data scope, finite
  timeouts/retry/call/token limits, cancellation and unknown-call reconciliation.
- Specify owned tags/fields, stable semantic identities, opt-in template migration,
  receipts and retained-field/manual ineligibility resolution. No CSV retirement.
  Define curated rule/rendered approval fingerprints, additional language-specific
  mnemonic approval when used, and presentation invalidation separately from linguistic
  acceptance. Support rule-only items and scope mnemonic changes to affected cards;
  supply rule-only, mnemonic-bearing and missing-language-variant payload examples.
- Supply valid/invalid payloads, migration/compatibility scenarios, check matrix
  by claim class and semantic comparison for legitimate alternatives. Identify
  consequential unresolved choices for the maintainer gate; do not approve them.

## Verification Notes

Inspect actual readers/public contracts and independently review examples. Resolve
every open checkpoint to a proposal or explicit decision. One bounded design pass;
unmet choices block approval. Preserve T-179 history. Evidence is design evidence,
not runtime or linguistic success.

## Implementation Notes

- Delivered proposal: `docs/v0.5.0-contract-proposal.md`, grounded in the
  actual strict claim/sidecar, phrase, authored, profile, corpus-fact,
  identity/CSV and managed-plan readers at base
  `61efcf45c49ddf4a66e29782ed99fa442e67c5d0`.
- Reviewed proposal SHA-256:
  `aef0af7771c125998b4e3d8e3ae5cdb72d23823ab9a150ed24a51d27fe1bc27b`.
  Sections 1–3 cover commands/profiles, schema shapes and explicit adapters;
  sections 3–4 cover full assertions, versions, origins and invalidation;
  sections 5–6 cover deterministic selection, batches, limits, permissions
  and interrupted calls; sections 7–9 cover ownership, receipts, curated
  approvals, payloads and compatibility vectors. Section 11 records the
  consequential unresolved choices for the maintainer, not their approval.
- No application code, executable schema, template, installed deck or
  runtime input changed. This bounded design pass does not implement the
  proposed commands or establish linguistic/runtime/native acceptance.
  T-179 history and the blocked T-186/T-205 gates remain unchanged.
- A dedicated code-simplifier clarified commands, schema upgrades,
  decision invariants and curated approval/invalidation requirements.
  Addition and review-fix readbacks preserved contract semantics.
- Independent conventional review used high-risk coverage: general
  code-review, schema/adapter integrity, security/permission/recovery,
  ownership/receipt/curated integrity, and separate comment/claims audit.
  With no nested coordinator facility, the implementing agent dispatched
  the independent lanes and relayed evidence to the same coordinator.
  Four boundary challenges plus an independent overall-outcome check
  narrowed the findings before the single fix round.
- Review dispositions, independently verified on the final hash:
  - O1 fixed (initially blocking): every existing-note Tags write needs
    observed managed reconciliation/approval or a manual preservation
    handoff; ordinary partial-sibling CSV rows remain withheld.
  - G1 fixed: cursors/deduplication use note-qualified CardRef pairs.
  - G2 fixed: explicit predecessor receipts distinguish replay/advance.
  - S1 fixed: canonical check inputs bind full current assertions;
    comparison binds both model outputs without breaking blindness.
  - S2 fixed: immutable legacy projection preserves actual human proof,
    overrides, corpus bindings and phrase combination/answer coverage.
  - SEC1 fixed: hosted permission is run-bound; renewal/resume retains
    consumed and unknown reservations. No proven exploit was asserted.
  - Minor selected-sense/evidence wording corrected. Duplicate identity
    findings consolidated; nullable provider ID was rejected because
    reconciliation already requires verified request identity. Registry,
    evaluation thresholds and native evidence remain gated decisions.
- Four scoped independent fix challengers and an overall verifier found
  all six findings fixed. Final coordinator verdict: approve as an
  unapproved design draft, zero remaining in-scope findings. This does
  not approve T-186, automatic policy, implementation or publication.
- Documentation checks: `git diff --check`, `taskrail validate` through
  `mise exec --`, and `uv run pytest -q tests/unit/install_docs_test.py`.
  All three documentation tests passed. Bounded structural inspection
  parsed all four JSON blocks, checked rule-only/mnemonic/missing-English
  and valid/invalid model shapes, table widths and the spec link.
  No new regression test is warranted for non-executable design prose;
  ruff/mypy/full pytest and runtime/native/model checks were not claimed.
  Future executable design vectors D1–D5 and M1–M9 remain consumer work.
- 2026-11-12T21:53:02Z: verification pass
- 2026-11-12T21:53:03Z: Completed design draft after independent review and documentation validation; proposal SHA256 aef0af7771c125998b4e3d8e3ae5cdb72d23823ab9a150ed24a51d27fe1bc27b. T-186 exact-revision approval and T-185 evaluation remain prerequisites, not granted here.
