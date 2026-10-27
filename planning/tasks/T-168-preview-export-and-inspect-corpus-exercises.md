---
id: T-168-preview-export-and-inspect-corpus-exercises
title: Preview, export and inspect corpus exercises
status: todo
priority: high
spec_ref: specs/v0.4.0.md#context-and-translation-recipes
dependencies:
    - T-164-render-the-corpus-v1-note-type-and-layout-content
    - T-166-rank-candidates-and-persist-sticky-selections
updated_at: "2026-10-27T23:10:48Z"
---

# T-168-preview-export-and-inspect-corpus-exercises Preview, export and inspect corpus exercises

## Description

Implements `corpus-cards preview`, `export` (initial creation only, via
`serialize_anki_csv` and `commit_csv_artifacts`) and `inspect [--check CSV]` per Fixed
Contracts **Commands**, with `nothing_to_export` and `card_drift`, the known-lemma deck
links in preview, and matrix rows.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Preview shows prompt, answer, citation, status, withholding reasons and each fact's
  claim status and producer, bounded by `--limit` (default 20, maximum 1000).
- Export writes only eligible selected exercises; an existing output exits 3 `exists`; no
  eligible exercise exits 1 `nothing_to_export`; withheld, stale and unreviewed content
  never reaches the CSV.
- `preview` exits 3 `invalid_reading` for an unknown reading ref; `preview`, `export`
  and `inspect` exit 4 `binding_drift` for a selection of another profile or language
  and 3 `profile_mismatch` for an unlisted or missing corpus, and report
  `target_missing`/`context_changed` selections as withheld with those reasons.
- `preview` with the known-lemma flags lists deck links; card fields never contain them.
- `inspect --check` passes on a fresh export and exits 1 `card_drift` on a tampered,
  stale or unreviewed CSV; it ignores `Profile`; no command writes files.

## Implementation Notes

- Start only when `taskrail status --json` shows active spec `specs/v0.4.0.md`; otherwise
  stop without `taskrail start` and report that v0.4.0 is not active.
- Follow-up rule (as in v0.3.0): (a) a defect in completed work this task builds on is
  fixed inside this task with a regression test; (b) follow-ups are filed only for work
  no remaining v0.4.0 task needs, with `--priority medium` or `high`, never `low`, and
  never with `--follow-up` on T-172; (c) a follow-up needing a maintainer decision (for
  example a Fixed Contracts disagreement) is created and then blocked with the reason
  "maintainer decision: ...".
- Register every new diagnostic code in `cli_contract.py` and add its row to the
  `docs/agent-cli-contract.md` table in this task (`cli_contract_test.py` compares them).

## Verification Notes

- Not started.
