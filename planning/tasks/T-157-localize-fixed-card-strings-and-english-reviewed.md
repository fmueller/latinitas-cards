---
id: T-157-localize-fixed-card-strings-and-english-reviewed
title: Localize fixed card strings and support English reviewed translations
status: todo
priority: high
spec_ref: specs/v0.4.0.md#german-and-english-output
dependencies:
    - T-156-add-the-language-table-and-english-terminology
updated_at: "2026-10-27T23:10:48Z"
---

# T-157-localize-fixed-card-strings-and-english-reviewed Localize fixed card strings and support English reviewed translations

## Description

Second half of `#german-and-english-output`. Move every fixed generated-card string
(principal-part answers in `cards.py` and `principal_relationships.py`, phrase prompts
and answers in `phrase_recipes.py`, form-parsing answer labels) into the
`(string_key, language)` table with `de` and `en` entries; `en` selects English, any
other subtag renders German exactly as today. Widen `Reference.language` and the
form-translation claim check (`claim_sidecar.py`) to `de`/`en`, matched to the profile's
primary subtag. Latin-first prompts such as `Formam resolve.` stay Latin.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- German goldens for principal-part, phrase and form-parsing preview and CSV output are
  byte-identical, including the non-`de` Latin-terminology profile.
- An `en` profile renders English fixed strings on every recipe with unchanged note IDs,
  card keys, slots and columns.
- Existing German phrase-reference and form-translation claim fingerprints are unchanged
  (golden claim sidecars still load and match).
- A German reviewed form translation or phrase reference never appears on an English
  card; an English one appears only after its own accepted decision.
- Switching a fixture deck profile from `de` to `en` makes its reviewed claims stale
  (documented consequence of the unchanged fingerprint contract).
- A `fr` form-translation or phrase-reference claim is rejected naming `de` and `en`.

## Implementation Notes

- Start only when `taskrail status --json` shows active spec `specs/v0.4.0.md`; otherwise
  stop without `taskrail start` and report that v0.4.0 is not active.
- Follow-up rule (as in v0.3.0): (a) a defect in completed work this task builds on is
  fixed inside this task with a regression test; (b) follow-ups are filed only for work
  no remaining v0.4.0 task needs, with `--priority medium` or `high`, never `low`, and
  never with `--follow-up` on T-172; (c) a follow-up needing a maintainer decision (for
  example a Fixed Contracts disagreement) is created and then blocked with the reason
  "maintainer decision: ...".

## Verification Notes

- Not started.
