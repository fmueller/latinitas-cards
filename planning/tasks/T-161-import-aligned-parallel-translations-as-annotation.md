---
id: T-161-import-aligned-parallel-translations-as-annotation
title: Import aligned parallel translations as annotation patches
status: todo
priority: medium
spec_ref: specs/v0.4.0.md#context-and-translation-recipes
dependencies:
    - T-154-capture-v0-4-0-characterization-goldens
updated_at: "2026-10-27T23:10:48Z"
---

# T-161-import-aligned-parallel-translations-as-annotation Import aligned parallel translations as annotation patches

## Description

Implements Fixed Contracts **Parallel translations**: `annotations
import-translations` emits a `latinitas-annotation-patch` of span-less `translation`
`add` operations from a normalized parallel corpus aligned by `native_ref`. It never
writes a sidecar or claim decision. Joins the CLI contract (envelope, matrix rows, codes
`unmatched_translation`, `unmatched_parallel`, `unsupported_language`; register a code
only if a sibling task has not already).

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- A parallel plain-text corpus normalized with the same `--citation-template` labels
  yields a patch that `annotations plan`, `approve` and `apply` accept with full
  provenance (producer kind/name/version, `input_hash`).
- Unmatched refs on both sides warn; a Latin or mismatched parallel header language
  exits 3 `unsupported_language`; identical existing output is a no-op, a different one
  exits 3 `exists`; timestamp rules follow v0.3.0.
- After apply and before `claims record`, no corpus card is eligible from these facts
  (asserted once the recipe tasks land, by the integrated validation task).

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
