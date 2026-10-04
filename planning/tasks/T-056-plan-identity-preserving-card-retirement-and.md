---
id: T-056-plan-identity-preserving-card-retirement-and
title: Plan identity-preserving card retirement and reactivation effects
status: todo
priority: high
spec_ref: specs/v0.2.0.md#note-and-card-lifecycle-contract
dependencies:
    - T-054-implement-bound-destination-snapshots-and-observed
updated_at: "2026-10-04T23:05:46Z"
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
