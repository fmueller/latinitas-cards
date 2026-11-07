---
id: T-220-specify-complete-morphology-coverage
title: Specify complete morphology coverage and transport parity
status: todo
priority: medium
spec_ref: specs/v0.5.5.md#planning-status-and-dependencies
dependencies: []
updated_at: "2026-11-07T17:09:08Z"
---

# T-220-specify-complete-morphology-coverage Specify complete morphology coverage and transport parity

## Description

Revise the inactive specs/v0.5.5.md and specs/v0.6.0.md planning package to require
complete noun/adjective morphology alongside verbs, with finite reference-backed
coverage, falsifiable acceptance and transport parity. Update specs/README.md.
This task tracks specification authoring, not future feature implementation.
Preserve the approved v0.5.1 plan and the active v0.5.0 spec and approval gates.

## Acceptance

- v0.5.5 defines noun declensions/cases/numbers/genders, adjective inflection,
  agreement and comparison, related pronoun/numeral/participial forms and verb
  parity. Distinguish lexical gender, contextual agreement and nonexistent cells.
- Require versioned references, independent expected-form and held-out cases,
  recipe-specific coverage reports and explicit schema/template upgrades.
- Missing evidence, ambiguity and defective paradigms never force invented
  forms or count as implemented coverage. Preserve translation-first defaults,
  optional production/clozes and bounded selection without an implicit card quota.
- v0.6.0 carries the same semantics/evidence/identities across file, managed,
  direct Anki and advertised enriched formats; incompatible destinations and
  lossy mappings fail explicitly, with native retry/migration acceptance planned.
- Proposed work packages map requirements to evidence owners, prerequisites,
  design approval, integrated acceptance and separate release permission.
- Independently review the spec contract and full bounded planning package;
  record candidate identity, findings and review budget. Validate Taskrail and
  whitespace without activating, starting or completing any tracked task.

## Verification Notes

- Run `mise exec -- taskrail validate` and `git diff --check`.
- Documentation-only: runtime morphology, native Anki and format round-trip
  scenarios are planned evidence, not checks executed by this authoring task.

## Implementation Notes

- Local authoring candidate at base 335190b3014ea415baa616e2a166dff7a1f06638:
  specs/v0.5.5.md, specs/v0.6.0.md, specs/README.md and this task. Taskrail
  regenerated planning/STATE.md for this task's count/timestamp. A reviewer
  inadvertently ran `taskrail next`, which also refreshed its advisory next
  action to T-184. No task was started. Active v0.5.0, existing task lifecycles
  and blocked maintainer gates are unchanged; STATE.md was not hand-edited.
- The task is isolated under inactive v0.5.5. Future implementation packages
  remain proposed within the specs; no implementation tasks are materialized.
- Review budget: two initial gate passes and at most two targeted rechecks
  across this revision. Gate 1 initial pass and targeted recheck 1 completed;
  Gate 2 initial pass and targeted recheck 2 passed with no blockers. Budget
  consumed: two initial passes and two targeted rechecks; none remain.
- G1-1 corrected: universal file fallback conflicted with inherited CSV
  retirement limits. v0.6.0 now distinguishes file/report handoff from applied
  lifecycle effects; declined-direct retirement stays pending/manual. The
  independent targeted recheck closed the blocker and passed Gate 1.
- Optional class/sense contrast examples remain P55-1 inventory work, not
  removed requirements. No outstanding Gate 1 blockers or scope decisions.
- Gate 1 reviewed spec/index SHA-256 values:
  - specs/v0.5.5.md: 3124a402515574b4d6b44e48b7d4787cba0758d2c26c30aedf3158253851e597
  - specs/v0.6.0.md: 096b54ae79f3837db2fb6e340237c502182b60b7b127faac594bf7d5f383f509
  - specs/README.md: ab9baa47ed278254226b7af8ea78837385e22fc6f4a446f3f7a14b0e01e38e0a
- Gate 2 reviewed the same three spec/index hashes and this task's description,
  acceptance and proposed-package boundary. No unowned requirement, dependency
  cycle or missing evidence owner was found in the bounded morphology plan.
- Final targeted recheck accepted the generated-state delta and receipt:
  planning/STATE.md SHA-256 is
  d511fc72e1bf8aeb3c279d4362e48cf5f11e03e51c5e6bd19aff05c71923b8f9.
  The task receipt at that review hashed to
  11814bf4475063233fa2b3df0d692862cd82d445b1b03570823c20c2409cb369;
  only this review-result/budget receipt changed afterward. Requirements,
  lifecycle fields and the three reviewed spec/index hashes are unchanged.
- Structural checks passed: `mise exec -- taskrail validate` reported
  `state valid`; `git diff --check` reported no whitespace errors.
- P55-1 is the first proposed design package after upstream contract approval;
  P60-M1 may then design parity before downstream implementation. Neither is
  claimed live-selector eligible here. Reference rights, exact inventories,
  thresholds, version mappings and native/model access remain explicit design
  or acceptance prerequisites; activation/execution/publication are not granted.
- On 2026-11-07, the maintainer approved the revised specification content at
  the three spec/index SHA-256 values recorded above, after an additional
  read-only reassessment found no material gap against the original request.
  This approves the local planning content only. Both v0.5.5 and v0.6.0 remain
  inactive; no implementation is authorized or started, no task lifecycle is
  changed, and future technical-contract and publication approvals remain
  separate. That planning approval alone did not authorize a commit or push.
- The maintainer subsequently authorized committing and pushing this approved
  documentation package on main, without activation or implementation.
