---
id: T-161-import-aligned-parallel-translations-as-annotation
title: Import aligned parallel translations as annotation patches
status: completed
priority: medium
spec_ref: specs/v0.4.0.md#context-and-translation-recipes
dependencies:
    - T-154-capture-v0-4-0-characterization-goldens
updated_at: "2026-10-29T18:41:27Z"
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

- Preserved the legitimate in-progress start across checkout transfer; the full
  17-file patch and source candidate hashes matched before continuation.
- Independent integrated QA passed on final candidate SHA256
  `0a6730b0de74cf31c67c834f0e916291c4fac66c285fc2b63a604ffc67fca582`:
  98 real CLI subprocesses, 360 assertions and 69 matrix/golden tests. German
  and English normalization, import, plan, approve, apply and individual claims
  acceptance preserved provenance and kept unaccepted facts ineligible.
- The scoped C8 simplifier made no changes. Independent disposition verification
  confirmed iterative surrogate-scan equivalence on 10,000 JSON-shaped values
  and structured errors instead of exit 70 for deep inputs. The changelog wording
  correction was independently read back; no required review findings remain.
- Prior F1-F6 findings were verified fixed. C3 is blocked medium-priority T-178;
  the spec's language rejection contract was not changed. A pre-existing approve
  receipt/default-sidecar path collision remains separate from T-161.
- 2026-10-29T18:41:22Z: verification pass
- 2026-10-29T18:41:27Z: Verified 2026-10-29T18:41:22Z: ruff, mypy and full pytest passed (3057 passed, 7 skipped); independent C8 review and integrated CLI QA passed. C3 recorded as blocked maintainer-decision T178.
