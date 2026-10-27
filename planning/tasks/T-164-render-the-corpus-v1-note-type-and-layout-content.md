---
id: T-164-render-the-corpus-v1-note-type-and-layout-content
title: Render the corpus-v1 note type and per-layout managed content fields
status: todo
priority: high
spec_ref: specs/v0.4.0.md#context-and-translation-recipes
dependencies:
    - T-163-derive-corpus-exercise-identities-and-candidates
    - T-157-localize-fixed-card-strings-and-english-reviewed
updated_at: "2026-10-27T23:10:48Z"
---

# T-164-render-the-corpus-v1-note-type-and-layout-content Render the corpus-v1 note type and per-layout managed content fields

## Description

Implements Fixed Contracts **Note type `corpus-v1`**: fields in order, four slots,
escaped Latin with the target span, citation and full passage, localized prompts and
answers with the `reference_guidance` string, its own templates and `CORPUS_CARD_CSS`
(`CORPUS_STYLE_VERSION = 1`, reference appearance classes for `light`/`dark`/`auto`;
`REFERENCE_CARD_CSS` and `REFERENCE_STYLE_VERSION` unchanged), the `corpus-v1` managed
layout, and the
per-layout content-field set used consistently by `managed_plans.py` plan and approve
checks and `destination_state.py` application.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Each recipe renders in `de` and `en` with a Latin-first front, full passage and
  citation on the answer, and `reference_guidance` on every translation card; exactly
  one slot is enabled and only when eligible.
- Controls and HTML are escaped (`validate_managed_controls` passes; a fixture with `<`,
  `&` and control characters is safe).
- Existing layouts' content-field sets, plans and goldens are unchanged; `corpus-v1`
  allows `Latin`, `Citation` and `Passage` updates, and a `Passage` change plans,
  approves and applies end to end on a synthetic snapshot.
- `CORPUS_CARD_CSS` renders light by default and dark under night-mode classes for
  `auto`; `REFERENCE_STYLE_VERSION`, `REFERENCE_CARD_CSS` and phrase package style checks
  are unchanged.
- The CSV columns follow the spec field order with `Language`, `Source Kind` `corpus`,
  `Note Schema`/`Generator` `corpus-v1`, `Profile` and `Source Scope`
  `corpus-passage-v1`; `Personal Notes` is never exported; tags use the transport
  directive.

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
