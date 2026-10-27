---
id: T-159-preserve-tei-notes-apparatus-and-editorial-markup
title: Preserve TEI notes, apparatus and editorial markup; fail closed on unknown elements
status: todo
priority: medium
spec_ref: specs/v0.4.0.md#tei-xml-adapter
dependencies:
    - T-158-normalize-perseus-epidoc-tei-structure-citations
updated_at: "2026-10-27T23:10:48Z"
---

# T-159-preserve-tei-notes-apparatus-and-editorial-markup Preserve TEI notes, apparatus and editorial markup; fail closed on unknown elements

## Description

Completes the **Text and markup** rules of Fixed Contracts **TEI subset** beyond the
elements T-158 finalizes: `bibl` and `desc` as notes; `app`/`lem`/`rdg`, `choice` handling and
standalone `reg`/`corr`/`expan`/`sic`/`abbr`/`orig`; the listed
text-keeping elements as `source_markup` spans; empty-span `cb`;
`gap` with a warning; unknown elements failing with names and counts unless
`--allow-unknown` keeps them in `source_extra`.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Fixtures for every listed element round-trip to goldens with expected canonical text,
  note kinds, headings and literal spans (independently written expected values, not
  regenerated from the implementation).
- Standalone `reg` and `sic` (Cicero/Caesar shapes) keep their words in `text`; inside
  `choice`, `sic`/`abbr`/`orig` leave `text` as `variant` notes.
- An unknown element and a drama `sp` exit 3 `unknown_element` listing names and counts;
  `--allow-unknown` keeps them in `source_extra`.
- `gap_in_text` warns; segmentation and `annotations segment` work on a TEI corpus.

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
