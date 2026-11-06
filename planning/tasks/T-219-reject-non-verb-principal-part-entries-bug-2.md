---
id: T-219-reject-non-verb-principal-part-entries-bug-2
title: Reject non-verb principal-part entries (BUG-2)
status: completed
priority: high
spec_ref: specs/v0.4.0.md#fixed-contracts
dependencies: []
updated_at: "2026-11-05T22:36:45Z"
---

# T-219-reject-non-verb-principal-part-entries-bug-2 Reject non-verb principal-part entries (BUG-2)

## Description

Fix BUG-2 in deck-first principal-part generation under
`specs/v0.4.0.md#fixed-contracts`. Positional layout matches must not enable
verb cards from noun, adjective or numeral dictionary entries.

## Acceptance

- Named non-verb examples and entries with implausible verb roles generate no
  notes and report `not_a_verb_entry`.
- Check the lemma and the configured infinitive, present first-person and
  perfect first-person roles without changing role order or note identity.
- Parenthetical notes cannot supply forms, alternatives or role boundaries;
  retain raw source evidence and withhold unbalanced notes.
- Setup can confirm an optional exact source-tag filter, persisted in the
  profile and enforced in generation without bypassing morphology.
- Existing valid German output remains unchanged.

## Verification Notes

- Red: all ten initial targeted regression cases failed on the unchanged
  implementation, including enabled non-verb notes and retained annotations.
- Additional red/green regressions cover annotated lemmas, confirmed hints,
  literal/encoded HTML attributes and comments, and mixed raw/encoded markup.
- Real CLI smoke: `generate --approve-scope` then `preview` on the three named
  patterns both report 3 source entries, 0 objects, 0 notes, 0 cards and three
  `not_a_verb_entry` skips; the CSV contains metadata/header only.
- Independent eligibility/setup review passed 112 focused tests including
  legacy goldens. Final conventional-review disposition recheck passed 114
  focused tests and resolved all findings; no required issues remain.
- Final mandatory chain passed: `uv run ruff check`, `uv run mypy` (199 source
  files), then `uv run pytest -v` (3259 passed, 7 skipped, 1 warning).

## Implementation Notes

- The report provides three named examples, not all seven original rows. Tests
  cover those three patterns plus additional non-verbs and invalid role values;
  verification of the exact original seven requires their source rows.
- The optional filter uses source tags, not a new field-mapping mechanism.
- Existing fixtures with `sum` and `esse` in the wrong configured role order
  were corrected; the escaping test now uses the gloss rather than an
  implausible perfect form. No characterization golden files changed.
- This prevents new nonsense exports; it does not remove previously imported
  cards from a user's Anki collection.
- Renumbered from T-206 to T-219 after rebasing onto parallel upstream work.
- The maintainer authorized committing and pushing the fix to main.
- 2026-11-05T22:36:45Z: verification pass
- 2026-11-05T22:36:45Z: Verified locally; no commit, push or publication requested
