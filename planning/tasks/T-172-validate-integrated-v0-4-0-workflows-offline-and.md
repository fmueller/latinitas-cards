---
id: T-172-validate-integrated-v0-4-0-workflows-offline-and
title: Validate integrated v0.4.0 workflows offline and prepare the 0.4.0 candidate
status: todo
priority: low
spec_ref: specs/v0.4.0.md#goals
dependencies:
    - T-170-document-corpus-first-tei-and-english-workflows
    - T-171-prepare-the-v0-4-0-native-acceptance-kit
    - T-159-preserve-tei-notes-apparatus-and-editorial-markup
    - T-161-import-aligned-parallel-translations-as-annotation
    - T-169-plan-managed-updates-for-corpus-exercises
    - T-167-deselect-and-restore-selected-corpus-exercises
updated_at: "2026-10-27T23:10:48Z"
---

# T-172-validate-integrated-v0-4-0-workflows-offline-and Validate integrated v0.4.0 workflows offline and prepare the 0.4.0 candidate

## Description

Delivery Order step 9. An offline end-to-end walkthrough using only documented CLI
commands in a temp dir outside the checkout, then the 0.4.0 candidate (version, lock,
changelog under `[Unreleased]` moved to 0.4.0, readiness doc, README). No tagging, release
or publication; no runtime code changes. Native evidence belongs to the gate.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- At start and before verify, `taskrail status --json` shows active spec
  `specs/v0.4.0.md` with only this task and the blocked gate open; otherwise block or
  requeue as T-124 did for v0.3.0.
- Walkthrough: TEI fixture and plain-text corpus normalize and validate; segmentation;
  `latinitas-morph/1` and translation patches (one via `import-translations`) through
  plan/approve/apply; `claims record` acceptances; `corpus-cards profile validate`,
  `select` with known lemmas, `preview`, `deselect`/`--restore`, `export`,
  `inspect --check`, `plan`; an English deck profile preview.
- Negative controls: applied but unreviewed translations give no eligible card; a passage
  correction gives `context_changed`; a tampered CSV fails `inspect --check`; an unknown
  TEI element exits 3; a `fr` corpus profile exits 3.
- The walkthrough also runs `corpus-cards select` and `preview` on the TEI corpus; shows
  that adding the known-lemma input changes only ranking values and report deck links
  (card fields and eligibility identical); and a test asserts no module under `src/`
  compares translation text for grading (no string equality on translation values
  outside rendering and validation).
- Characterization goldens from T-154 are unchanged since the commit that added them.
- TDD: update the release-metadata test first (RED on the old version), then set
  `project.version` and the `uv.lock` entry to 0.4.0, move `[Unreleased]` entries under
  `## [0.4.0] - YYYY-MM-DD` with comparison links, and add `docs/release-v0.4.0.md`
  (scope, walkthrough, limits, native evidence pending the gate). T-173 may correct only
  the date before tagging.
- Run `UV_OFFLINE=1 bash scripts/check-dist.sh` and `bash scripts/check-install.sh
  "$(uv python find 3.14)"` as T-124 did (rerun without `UV_OFFLINE` if only the cache
  is missing; record "not run: offline" if no network).
- No tag, release, upload, workflow dispatch, spec activation or runtime change. A
  product defect found here is filed with `taskrail task new --spec-ref
  specs/v0.4.0.md#<anchor> --priority high --title "..."` (never `--follow-up` on this
  task), then this task is requeued with `taskrail block <id> --reason "requeued behind
  T-NNN"` and `taskrail unblock <id> --reason "requeued behind T-NNN"`; uncommitted
  candidate edits are discarded first.
- All goldens and the full mandatory chain pass; `mise run check` and `taskrail validate`
  pass.

## Verification Notes

- Not started.
