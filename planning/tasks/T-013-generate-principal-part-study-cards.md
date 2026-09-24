---
id: T-013-generate-principal-part-study-cards
title: Generate principal-part study cards
status: completed
priority: high
spec_ref: specs/v0.1.0.md#principal-part-card-generation
dependencies:
    - T-001-define-deck-profile-contract
    - T-002-stable-generated-note-identity
    - T-003-parse-principal-parts
    - T-011-validate-the-representative-deck-and-german
updated_at: "2026-09-24T20:42:49Z"
---

# T-013-generate-principal-part-study-cards Generate principal-part study cards

## Description

Generate Latin-first principal-part completion and recognition exercises from successfully
parsed lexical entries. Preserve semantic roles, source provenance, stable identity, and
the configured German gloss while carrying parser failures forward as structured skips.

## Acceptance

- Completion exercises omit one named principal part and answer with the missing form and
  semantic role.
- Recognition exercises map a supplied principal part to its lemma, complete principal
  parts, role or stem, and configured German gloss.
- Each exercise is an independent generated note by default and uses the stable identity
  and ownership contracts.
- German-to-Latin production is not generated.
- Incomplete, unsupported, and ambiguous parser results become structured skips without
  guessed cards.
- Only recipes selected in the confirmed profile run; compatibility suggestions do not
  enable recipes implicitly.
- Tests cover both recipes, every supported parser layout, provenance, skips, wording
  changes that preserve identity, and semantically distinct exercises that do not collide.
- Implement with red/green TDD in focused modules rather than expanding `cli.py`.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- 2026-09-24T20:42:35Z: verification pass
