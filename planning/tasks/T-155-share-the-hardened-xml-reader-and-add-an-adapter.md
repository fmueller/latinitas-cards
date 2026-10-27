---
id: T-155-share-the-hardened-xml-reader-and-add-an-adapter
title: Share the hardened XML reader and add a corpus adapter registry
status: todo
priority: high
spec_ref: specs/v0.4.0.md#tei-xml-adapter
dependencies:
    - T-154-capture-v0-4-0-characterization-goldens
updated_at: "2026-10-27T23:10:48Z"
---

# T-155-share-the-hardened-xml-reader-and-add-an-adapter Share the hardened XML reader and add a corpus adapter registry

## Description

Refactor only (red/green TDD per AGENTS.md). Move the expat-based reader now private
in `usfx_adapter.py` (`_EventReader`: DOCTYPE/entity rejection, 64 MiB and depth-64
limits, streaming SHA-256) into a shared module with optional namespace processing and
adapter-neutral diagnostic messages. Replace the hard-coded adapter `if` chain in
`commands/corpus.py` with an explicit registry of adapter name to builder, keeping
`usfx` and `text` behaviour, option checks and exit codes unchanged. The TEI builder is
registered by a later task.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- USFX and plain-text normalization, legacy Vulgate goldens, characterization goldens
  and the full-Vulgate oracle are byte-identical.
- The shared reader is used by the USFX adapter; a test proves DTD, entity, depth-65 and
  over-64-MiB rejection through the shared module with namespace processing on and off.
- An unknown `--adapter` still exits 2; the registry is the only adapter dispatch point.

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
