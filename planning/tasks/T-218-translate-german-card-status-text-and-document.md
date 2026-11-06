---
id: T-218-translate-german-card-status-text-and-document
title: Translate German card status text and document managed refresh
status: completed
priority: high
spec_ref: specs/v0.4.0.md#german-and-english-output
dependencies: []
updated_at: "2026-11-06T22:42:54Z"
---

# T-218-translate-german-card-status-text-and-document Translate German card status text and document managed refresh

## Description

Fix BUG-1 under `specs/v0.4.0.md#german-and-english-output`: translate German
generated-card status text and generated phrase-review reasons without changing
English defaults, reviewer-authored prose, machine statuses or linguistic decisions.

## Acceptance

- Render every controlled German terminology and fixed card-string key, reject
  English status markers and allow identical German/English terms only explicitly.
- Localize unresolved/review-required markers, reference headings and generated
  phrase reasons, preserving identity, fingerprints, card eligibility and authored text.
- Document the managed plan/approve/emit/observe sequence and recovery-only reconcile
  for imported notes, retaining the existing native-client and preservation limits.

## Verification Notes

- Red/green: the all-key check failed on the original English `role_unresolved`;
  the German stale-phrase regression failed before reason localization. Both now pass.
- Final implementation chain: ruff passed; mypy passed for 199 source files;
  pytest passed with 3235 passed, 7 skipped and one existing negative-test serializer
  warning (394.57 seconds). Ruff format checked all 499 files; diff check passed.
- Independent review approved with no findings, independently passed 107 focused
  tests and verified all 171 golden manifest hashes. Simplification made no edits.
- Temporary old/new generation and managed roundtrip preserved note ID, eligibility,
  cards/history and supported six prompt/answer updates through plan/approve/emit/
  observe; replanning was no-write. This is synthetic offline evidence, not native
  scheduling certification or an import into the reported collection.
- Chromium inspection of completion/recognition and accepted/stale/mismatched phrase
  states found legible, unclipped German generated text, with no English status
  markers in the inspected answers. This is not native Anki UI certification.
- Initial validation exposed old English assertions and frozen German goldens.
  The maintainer explicitly approved the targeted golden exception and generated
  phrase-reason scope; the spec records the exception. Targeted harness replay and
  independent byte comparison confirmed only three phrase outputs changed, solely
  by `Reviewed reference` to `Geprüfte Referenz` substitution (JSON-escaped in preview).
  Only their three CAPTURE hashes changed; inputs, sidecars, original capture-commit
  provenance and legacy Vulgate outputs remain unchanged.

## Implementation Notes

- Nine existing German strings were translated. `assess_claim` keeps an English
  default for existing callers; phrase presentation explicitly selects card language.
  Applicable reviewer reasons stay verbatim; stale decisions use generated reasons.
- Invalid-reference technical details remain in diagnostics, not German card answers.
  Reconciliation accepts current anchors and abandons pending effects for reviewed
  recovery; it is not an unconditional success step or permission to replay a CSV.
- During delivery, remote main introduced a different T-206. This work was rekeyed
  to T-218 without changing the reviewed implementation or the remote planning work.
  The initial local T-206 verification passed at 2026-11-05T22:31:22Z; this task
  supersedes that local identifier. No user collection was imported.
- 2026-11-06T22:42:54Z: verification pass
- 2026-11-06T22:42:54Z: Implemented and reviewed BUG-1; rekeyed from local T-206 to avoid remote task collision. Ruff/mypy passed after documentation-only fast-forward; full implementation suite passed 3235 tests with 7 skips.
