---
id: T-056-plan-identity-preserving-card-retirement-and
title: Plan identity-preserving card retirement and reactivation effects
status: completed
priority: high
spec_ref: specs/v0.2.0.md#note-and-card-lifecycle-contract
dependencies:
    - T-054-implement-bound-destination-snapshots-and-observed
updated_at: "2026-10-05T03:16:38Z"
---

# T-056-plan-identity-preserving-card-retirement-and Plan identity-preserving card retirement and reactivation effects

## Description

Represent card-level lifecycle effects without claiming the selected CSV transport can perform structural changes. Consume the existing coherent-object identity, frozen template registry, and conditional-card rules rather than rebuilding completed foundations.

## Acceptance

- Plan retain, eligible add, retire, and explicit reactivate effects independently for stable semantic card/template bindings. Mutable content, enabled recipes, profile/theme/software changes never re-key surviving notes/cards.
- Multiple objects/senses within one source note retain separate identities; equal visible lemmas across entries do not merge. Ambiguous splits/reconciliation require confirmation.
- Missing, ambiguous, withheld, inapplicable data or parser failure cannot cause invalid additions or implicit retirement/deletion. A noun does not acquire verb cards.
- Retirement retains last safe content/provenance without labeling it newly approved; block changes if the schema cannot represent retention. Card retirement leaves siblings active; whole-object retirement is distinct and uses latinitas::retired only with the appropriate approved lifecycle effect.
- Record pre-retirement suspension and tool-owned provenance. Reactivation requires eligibility, explicit approval, and existing identities; user-suspended and uncertain-ownership cases never authorize unsuspension.
- Structural CSV apply effects remain unsupported, including enabling guard fields that would indirectly add cards or clearing fields that would remove them. No tags or empty content approximate suspension.
- Fixtures cover sibling eligibility changes, two senses, tool-retired versus user-suspended cards, unknown ownership, and re-enablement without approval. Include a formerly absent slot becoming eligible during a sibling content update and an enabled slot whose destination card is missing; eligibility is not proof of actual destination card existence. Refuse payloads that change the card set under content-only scope.

## Verification

Use red/green tests and the mandatory ruff, mypy, pytest -v chain. Native history preservation remains a separate transport evidence gate.

## Workflow-v3 implementation evidence

- Step 1: fetched origin/main and confirmed accepted T-055 base containment;
  clean HEAD matched accepted upstream before any writer. Taskrail validate and
  next JSON selected exactly this task under the pinned v0.2.0 lifecycle heading,
  with completed T-054 dependency. No stale local main or timestamp drift consumed.
  Read full task/spec, T-048 transport contract, T-054/T-055 and existing identity,
  cards, notes and checkpoint ownership contracts. One cycle only.
- Step 2: added focused offline card_lifecycle planner using GeneratedNote,
  frozen registry, actual bound destination rows and observed baseline anchors.
  Independent retain/add/retire/explicit-reactivate effects retain existing IDs.
  Retention is a stale/pending whole-note fields/provenance bundle, not a newly
  approved claim. No implicit lifecycle action on incomplete/uncertain proposals.
  Explicit object retirement is distinct from per-card retirement; neither can
  apply through CSV. Observed retirement receipts reuse existing anchor card rows,
  not a competing source of truth. No CLI growth, native apply or transport widening.
- RED: initial focused suite failed with ModuleNotFoundError for card_lifecycle.
  GREEN: 15 lifecycle tests passed. Additional asymmetric RED cases failed with
  ['retire', 'add'] versus ['retire'] and retired-tag-only DID NOT RAISE;
  fixes excluded new eligibility during whole retirement and blocked reserved
  lifecycle tag changes in the journal. Existing destination-state fixtures now
  explicitly bind actual frozen slots rather than opaque synthetic card rows.
- Step 3: initial exact ruff/mypy/pytest -v chain passed: 78 checked files,
  757 tests. Focused lifecycle/destination-state run passed 53 tests.
- Step 4: dedicated Task loaded code-simplifier; no changes recommended, no
  suggestions rejected. Its ruff, mypy and focused 53-test checks passed.
- Step 5: separate parallel independent Tasks loaded code-reviewer in General,
  Security, Database/persistence and Python lanes. General loaded ECC code-reviewer
  and common rules; Security loaded security-reviewer/security-review; Database
  loaded database-reviewer/postgres-patterns/database-migrations; Python loaded
  python-reviewer/python-patterns. General is mandatory, Security covers ownership
  and data-loss boundaries, persistence covers observed anchors/journal, Python
  covers changed language and test/error semantics. No framework/ML/native lanes
  apply. Three specialist lanes meet the soft budget.
  Security and Python: "No concrete task-relevant findings."
- Fresh candidate validation confirmed F1 and deduplicated DB-1 under it; no
  candidates rejected. F1, verbatim: "Reject managed-set membership that mixes a
  legacy unkeyed member with keyed objects for the same `(scope, source_id)`;
  that legacy identity cannot be unambiguously assigned when the source is split
  into objects." Evidence: destination_state.py membership tuple duplicate check.
  DB-1, verbatim: "Reject mixed legacy and object-key membership for the same
  source entry; the legacy member has no object key to disambiguate it from the
  separately keyed senses."
- Step 6: F1/DB-1 fixed with source-pair keyed/unkeyed validation. Two ordering
  tests RED with DID NOT RAISE, then GREEN (2 passed). All-keyed senses and
  legacy-only observed-state regressions remain green. No deferrals/followups.
- Step 7: full exact chain passed (ruff, mypy 78 files, pytest -v 759 passed),
  git diff --check passed. Fresh read-only disposition-verification Task loaded
  General/Python reviewer guidance, Python companion and common rules; independently
  repeated full gates and focused 55 tests. Conclusions: "F1 — RESOLVED."
  "DB-1 — RESOLVED." "No newly introduced task-relevant findings."
  One review/fix/recheck cycle; no remaining findings.
- Acceptance fixtures include sibling changes, two senses with equal lemmas, noun
  no-verb eligibility, uncertain proposals, user/tool/unknown suspension ownership,
  re-enable without approval, formerly absent eligible slot during sibling update,
  enabled slot missing actual card, guard creation/front clearing/tag approximation,
  whole-object versus single-card retirement and unrepresentable safe retention.
- Remaining scoped work is existing later tasks: managed CLI/application and
  native full-card/history evidence. T-053 presentation remains pending; no second
  task selected or started and no release readiness/native safety claim is made.

## Implementation Notes

- 2026-10-05T03:16:38Z: verification pass
- 2026-10-05T03:16:38Z: Workflow-v3 steps 1-7 complete with strict RED/GREEN, dedicated simplifier, General/Security/Database/Python reviews, deduplicated F1/DB-1 fixed, fresh independent disposition verification and final mandatory gates. Offline lifecycle only; no native apply/proof.
