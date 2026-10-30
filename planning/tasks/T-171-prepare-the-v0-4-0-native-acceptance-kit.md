---
id: T-171-prepare-the-v0-4-0-native-acceptance-kit
title: Prepare the v0.4.0 native acceptance kit for Linux Anki Desktop
status: completed
priority: medium
spec_ref: specs/v0.4.0.md#native-acceptance
dependencies:
    - T-164-render-the-corpus-v1-note-type-and-layout-content
    - T-157-localize-fixed-card-strings-and-english-reviewed
    - T-168-preview-export-and-inspect-corpus-exercises
updated_at: "2026-10-30T01:39:22Z"
---

# T-171-prepare-the-v0-4-0-native-acceptance-kit Prepare the v0.4.0 native acceptance kit for Linux Anki Desktop

## Description

Offline preparation for `#native-acceptance`; it records no native evidence itself.
Build an isolated synthetic kit: `corpus-v1` note-type setup instructions from its
templates and `CORPUS_CARD_CSS`, a German and an English corpus CSV covering all four
recipes, an update CSV, and English principal-part, phrase and form-parsing samples,
plus a checklist matching the spec (import, re-import with first-field matching and
`Personal Notes` unmapped, front/answer in light/dark/auto with a light-night-light
toggle). Follow the v0.2.1 native verification documents' structure.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- The kit regenerates deterministically from committed fixtures with documented commands.
- The checklist covers every item in `#native-acceptance` and states that AnkiMobile is
  unverified; no task claims native evidence.

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
- 2026-10-30T01:39:22Z: verification pass
- 2026-10-30T01:39:22Z: Offline preparation only. Dedicated simplification, independent medium-risk coordinated review and adversarial challenge, independent two-environment acceptance QA and full ruff/mypy/pytest chain passed. Native Linux Anki Desktop and AnkiMobile remain unverified; no release authorization. Existing maintainer decisions remain untouched; do not start T-172 until its readiness condition is met.
