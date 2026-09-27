---
id: T-035-verify-the-revised-note-architecture-before
title: Verify the revised note architecture before publication
status: todo
priority: high
spec_ref: specs/v0.1.0.md#revised-architecture-release-gate
dependencies:
    - T-029-model-coherent-learning-objects-with-stable-note
    - T-030-generate-conditional-sibling-cards-with-stable
    - T-031-require-evidence-backed-field-suggestions-and
    - T-032-review-extraction-coverage-and-correct
    - T-033-publish-reference-multi-card-templates-and-safe
    - T-034-guard-legacy-note-model-transitions-with-explicit
    - T-036-preserve-csv-and-manifest-recovery-on-interruption
updated_at: "2026-09-27T08:56:30Z"
---

# T-035-verify-the-revised-note-architecture-before Verify the revised note architecture before publication

## Description

Establish release-candidate evidence for the revised model. Old completed readiness and
GUI retests remain truthful but do not approve this candidate. T-014 remains blocked
until this gate passes and the owner approves the exact candidate for publication.

## Acceptance

- Synthetic native GUI first/update/no-op imports show one object note with multiple
  sibling cards, distinct source objects, source immutability, stable provenance,
  shared Personal Notes/tags, and deterministic regeneration.
- Structural model tests explicitly assert one coherent object maps to one note with
  multiple card keys/ordinals; independent sources stay independent, profile changes
  retain object identity, conditional cards require normalized content, and source
  fields/templates/deck structure remain unchanged. Native tests verify the same shape;
  no FSRS simulation or note-level schedule is introduced.
- Require T-036 interruption recovery, T-029 independent-manifest identity/vector tests,
  T-032 normalized eligibility regressions and T-033 external transport/schema assertions
  to pass before publication. Existing successful tests do not disprove these probes.
- Add a supported card while preserving note ID and surviving card IDs/history. Exercise
  missing-form, missing-prior-evidence, and profile-disable review paths safely.
- Compare all card/review-log columns on compatible updates and all note/card/review-log
  columns on no-op. Card additions permit only expected new rows, preserving old rows.
- Inspect actual front/back rendering for both recipes and missing forms; reject black
  captures. Verify shared note IDs/distinct ordinals and independent card states; observe
  configured sibling burying without simulating FSRS.
- Verify tag replacement warning and actual limitation separately from Personal Notes
  protection. Record Anki version/platform, sample limits and fresh-start rehearsal.
- Run ruff, strict mypy, full pytest, supported Python CI and Taskrail validation; review
  candidate docs/notices/version/changelog. Obtain explicit exact-candidate approval
  before unblocking T-014 through CLI. Completion does not itself authorize publication.

## Verification Notes

- Record executed commands, decisive results, inspected visuals and limits. Cite verify
  timestamps rather than ignored artifact paths. No tests have run for this planned gate.

## Implementation Notes
- T-026/T-027/T-028 remain closed. This tests changed architecture, not resolved process
  defects. General provisioning and broad compatibility remain outside v0.1.0.
