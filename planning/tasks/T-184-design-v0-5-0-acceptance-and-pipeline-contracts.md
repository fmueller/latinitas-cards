---
id: T-184-design-v0-5-0-acceptance-and-pipeline-contracts
title: Design v0.5.0 acceptance and pipeline contracts
status: todo
priority: high
spec_ref: specs/v0.5.0.md#open-technical-design-checkpoints
dependencies: []
updated_at: "2026-11-02T23:10:11Z"
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
