---
id: T-156-add-the-language-table-and-english-terminology
title: Add the generated-content language table and English terminology
status: completed
priority: high
spec_ref: specs/v0.4.0.md#german-and-english-output
dependencies:
    - T-154-capture-v0-4-0-characterization-goldens
updated_at: "2026-10-29T03:46:56Z"
---

# T-156-add-the-language-table-and-english-terminology Add the generated-content language table and English terminology

## Description

Implements the terminology half of `#german-and-english-output` and Fixed Contracts
**Languages** and the controlled vocabulary used by **`latinitas-morph/1`**. Replace the
two-element Latin/German tuples in `terminology.py` with tables keyed by `la`, `de`, `en`;
add English for every feature name, value and principal-part role label; add `degree`
and `verbform` values in all three languages. Profile validation accepts `user`
terminology for primary subtag `de` or `en` and rejects others with the existing error
naming the supported tags; other tags with Latin terminology keep validating and render
German fixed text. Fixed card strings are localized by the next task.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Every terminology key renders in `la`, `de` and `en`; a test proves `en` has an entry
  for every `de` key and vice versa.
- German characterization goldens stay byte-identical; an `en` profile in `user` mode
  renders English labels with unchanged note IDs, card keys and columns.
- `user` terminology with `fr` fails profile validation naming `de` and `en`; a `fr`
  profile with Latin terminology still validates and renders German output.
- `terminology_report` reports the effective language for `en`.

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
- 2026-10-29T03:46:51Z: verification pass
- 2026-10-29T03:46:56Z: Verify run 2026-10-29T03:46:51Z. Left for T-157: front role labels (cards.role_display_label), fixed German card strings, and the --terminology help text 'user (German)' pinned by the cloze-jsonl help goldens.
