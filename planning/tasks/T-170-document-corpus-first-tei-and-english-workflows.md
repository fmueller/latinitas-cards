---
id: T-170-document-corpus-first-tei-and-english-workflows
title: Document corpus-first, TEI and English workflows and complete the contract registry
status: todo
priority: medium
spec_ref: specs/v0.4.0.md#fixed-contracts
dependencies:
    - T-159-preserve-tei-notes-apparatus-and-editorial-markup
    - T-161-import-aligned-parallel-translations-as-annotation
    - T-169-plan-managed-updates-for-corpus-exercises
    - T-157-localize-fixed-card-strings-and-english-reviewed
updated_at: "2026-10-27T23:10:48Z"
---

# T-170-document-corpus-first-tei-and-english-workflows Document corpus-first, TEI and English workflows and complete the contract registry

## Description

User and agent documentation for every v0.4.0 feature: corpus profile, recipes and
review flow, parallel translation import, ranking and selection reasons, deselect,
export/adoption/managed updates, TEI subset with supported elements and exclusions,
English output and the language-switch consequence. Reconcile
`docs/agent-cli-contract.md` with the registry, matrix and schemas; update README and
`specs/README.md` wording where it describes v0.4.0.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- The diagnostic registry equals the documented table; every `corpus-cards` command
  has matrix rows exercised by the contract tests; schemas match `docs/schemas/`.
- Docs list every recognised TEI element and every outcome/withholding reason exactly
  as in the schemas (a test compares the reason vocabulary).
- Docs state that reference translations are guidance, selection is sticky, CSV cannot
  create or retire notes, and a deck language switch stales claims.

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
