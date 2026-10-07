---
id: T-073-decide-representative-deck-native-acceptance
title: Decide representative-deck native acceptance for morphology disclosure
status: completed
priority: high
spec_ref: specs/v0.2.0.md#morphology-themes
dependencies: []
updated_at: "2026-10-07T21:07:40Z"
---

# T-073-decide-representative-deck-native-acceptance Decide representative-deck native acceptance for morphology disclosure

## Description

Found by the T-069 review. specs/v0.2.0.md Delivery Order item 5 asks for AnkiMobile and Desktop
acceptance on a sanitized representative deck before marking disclosure and contrast compatible.
T-062 used a synthetic four-verb kit, captured no native screenshots for the 2026-10-06 run
although its acceptance lists them, and did not exercise a CSS change on an existing scheduled
note type. Needs a maintainer decision. Release-blocking for T-068.

## Acceptance

- Either run the native kit on the sanitized representative deck (tests/fixtures/representative-university-latin.apkg) with screenshots, or record a dated maintainer waiver in specs/v0.2.0.md, T-062 notes, and docs/morphology-native-verification.md.
- docs/morphology-native-verification.md states exact client/OS versions or explicitly notes what was reported only by the maintainer.
- User-facing docs describe disclosure as verified only within the recorded scope.

## Verification Notes

- Step 1: fetched origin/main, confirmed accepted baseline
  `a9ae3c7dfdf700d4910ee00a8bda81101ad0809b` is contained and matches HEAD;
  clean initial worktree, no transfers. `taskrail validate` returned `state valid`;
  `taskrail next --json` selected exactly T-073 with `off_spec: false`. Active
  spec is pinned to `specs/v0.2.0.md`, all-open scope, medium sequential worker.
  No active owner before starting only this task through Taskrail.
- Read current task, T-062, native verification, complete active spec, reference
  presentation disclosure and CLI help. Source-thread read confirmed Felix Müller
  (`fmezza`) wrote “I waive the addtional deck gate, continue” on 2026-10-07
  (20:57:20 UTC, 22:57:20 Europe/Berlin). No new native run or product change.
- Step 2: documentation-only acceptance assertions, not production RED/GREEN.
  Before editing, a Python assertion for the dated exact waiver in all three
  required records failed at `specs/v0.2.0.md`. After editing, assertions passed
  for the date/source/quote, narrow waiver, reported client/OS provenance,
  missing screenshots/representative run/scheduled CSS proof, scoped disclosure
  and static fallback. Native matrix and bound hashes were retained.
- Step 3: initial exact chain passed: `uv run ruff check`, `uv run mypy`
  (87 source files), `uv run pytest -v` (844 passed). `git diff --check` passed.
- Step 4: dedicated Task loaded `code-simplifier`, inspected actual diff and
  factual scope, made no changes. Its scope assertions and whitespace check
  passed; no simplification suggestions were rejected.
- Steps 5–6: dedicated read-only General Task loaded `code-reviewer`, reviewer
  index and General ECC guidance; returned verbatim "No concrete task-relevant
  findings." General has no companion skills. Python/framework/security/database
  and other specialist lanes omitted: Markdown/lifecycle-only scope, no code,
  API, schema, runtime, trust-boundary or storage behavior changes. Fresh
  candidate-validation Task loaded `code-reviewer` and returned verbatim "No
  concrete task-relevant findings." No candidate IDs, rejected/deduplicated
  candidates, fixes or deferrals. Reviewers inspected the actual diff but relied
  on caller source-thread verification and chain results; their shell PATH
  lacked Taskrail. Caller `taskrail validate` passed with the configured PATH.
- Step 7: fresh full exact chain passed: ruff, mypy (87 source files), pytest -v
  (844 passed). Final documentation assertions passed, including unchanged
  historical observation/kit tables and hash rows, exactly six expected
  Markdown/lifecycle files and no ignored artifact references. Whitespace and
  Taskrail validation passed. Fresh disposition-verification Task loaded
  `code-reviewer`, independently reran the chain (844 passed), assertions and
  Taskrail validation, and returned verbatim "No concrete task-relevant
  findings." It checked the source-thread decision and unchanged generated
  reference block. One review cycle, no unresolved findings or deferrals.
- Step 8: Taskrail verification at 2026-10-07T21:07:40Z recorded pass and
  `taskrail complete` recorded completion after these review gates. Validation
  returned `state valid`, with zero active tasks and no blockers. Read-only
  `next --json` selects T-068 (`off_spec: false`); no second task is started
  and no release is authorized here. Future presentation changes still require affected
  native retests. No new follow-up was discovered; existing pinned tasks remain
  with the orchestrator, and off-spec T-038/T-064–T-067 are excluded.

## Implementation Notes

- Waiver branch only: spec, T-062 notes and native verification record the dated
  additional representative-deck decision; reference-note-type disclosure is
  limited to the recorded 2026-10-06 synthetic four-verb/client scope. No source,
  tests, fixture, native kit, screenshot or release readiness changes. Exact
  client/device/OS values are explicitly maintainer-reported; unreported Desktop
  OS and mobile patch/build values are not invented.
- 2026-10-07T21:07:40Z: verification pass
- 2026-10-07T21:07:40Z: Record 2026-10-07 maintainer waiver only of additional representative-deck native gate; preserve dated synthetic evidence, provenance, limits and static fallback. Reviewed and validated docs-only cycle.
