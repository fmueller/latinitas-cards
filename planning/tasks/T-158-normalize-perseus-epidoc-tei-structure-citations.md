---
id: T-158-normalize-perseus-epidoc-tei-structure-citations
title: Normalize Perseus/EpiDoc TEI structure, citations and header metadata
status: todo
priority: medium
spec_ref: specs/v0.4.0.md#tei-xml-adapter
dependencies:
    - T-155-share-the-hardened-xml-reader-and-add-an-adapter
updated_at: "2026-10-27T23:10:48Z"
---

# T-158-normalize-perseus-epidoc-tei-structure-citations Normalize Perseus/EpiDoc TEI structure, citations and header metadata

## Description

Implements the structural part of Fixed Contracts **TEI subset**: register `tei` in
the adapter registry; parse one namespaced TEI P5 file with the shared hardened reader;
require one Latin edition division; derive passages (textparts and passage `l`, `lg`
transparent, notes/deletions between lines attached as specified, empty passages skipped
with a warning), `tei-cts` citations, `native_ref` encoding, work ID and header metadata
with descriptor override; extend the corpus schema (`adapter` gains `tei/1`, `notes.kind`
gains `note` and `variant`) and regenerate schemas. This task finalizes, with their spec semantics, `TEI`, `teiHeader`
and the header elements read, `text`, `body`, `div`, `p`, `ab`, `l`, `lg`, `head`,
`note`, `del`, `lb`, `pb` and `milestone`. T-159 owns every other text and markup rule;
elements it owns may fail as `unknown_element` here, so this task's fixtures use only
the elements above.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Synthetic prose (book/chapter/section with `section/p`) and verse (book/line with `l`
  directly in the textpart and in `lg`, a between-line `note` holding an `l[@n]` and a
  `del`, an all-`del` line) fixtures normalize to committed goldens that pass `corpus
  validate`.
- Citation cases: `pr`, `3a`, `62b` and percent-encoded components; duplicate citation
  exit 3 `duplicate_source_ref`; inconsistent unit paths and stray verse-textpart text
  exit 3 `invalid_structure`; a `cRefPattern` label mismatch (`line, Book`) warns
  `citation_mismatch` only.
- Missing edition (exit 3 `invalid_structure`), non-Latin edition (exit 3
  `unsupported_language`), translation division (`skipped_division` warning), wrong
  namespace, DTD, entity, depth-65 and over-64-MiB inputs (exit 3).
- Work ID from `--work`, edition URN, filename `idno` and `corpus_id` fallbacks; header
  fields incl. the `sourceDesc//title` fallback; `descriptor_override` warnings.
- `--manifest`, approval flags and `--citation-template` with `--adapter tei` exit 2.
- New codes are registered with their exit classes and documented; schema drift test
  passes; USFX/plain-text goldens and the full-Vulgate oracle are byte-identical.

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
