---
id: T-037-specify-early-deck-based-phrase-and-grammar
title: Specify early deck-based phrase and grammar practice
status: completed
priority: medium
spec_ref: specs/v0.1.0.md#roadmap-to-v10
dependencies: []
updated_at: "2026-10-02T16:40:36Z"
---

# T-037-specify-early-deck-based-phrase-and-grammar Specify early deck-based phrase and grammar practice

## Description

Document the agreed early deck-based phrase and grammar practice release in
specs/v0.2.1.md and the roadmap in specs/v0.1.0.md#roadmap-to-v10. This task
authors the planned specification only; it does not implement the feature or
activate a new release spec.

## Acceptance

- Add a planned, inactive v0.2.1 spec between form analysis and corpus normalization.
- Specify natural combinations of existing deck vocabulary, supported inflectional
  variants, separate translation and grammatical-recognition recipes, ambiguity handling,
  reviewed output, and reuse of stable identity and export contracts.
- Update the spec index, roadmap, and v0.4/v0.5 boundaries to distinguish the early
  deck-first feature from later corpus generation and broader grammar exercises.
- Keep v0.1.0 active and leave application code unchanged.

## Verification Notes

- Run Taskrail validation and inspect the spec list and release boundaries.
- Run the ruff/mypy/pytest chain before recording verification and completion.

## Implementation Notes

- Feature implementation and its acceptance tests remain future work; completion of
  this task means only that the specification and roadmap have been authored.
- 2026-10-02T16:40:36Z: verification pass
- 2026-10-02T16:40:36Z: Specification and roadmap authored only; v0.2.1 feature implementation remains future work. Mandatory checks passed before verification.
