---
id: T-003-cli-extraction
title: Move command implementations and helpers out of cli.py
status: todo
priority: medium
spec_ref: specs/v0.1.0.md#module-extraction
dependencies: []
updated_at: "2026-09-14T09:14:51Z"
---

# T-003-cli-extraction Move command implementations and helpers out of cli.py

## Description

`e7806ca` created `src/latinitas_cards/commands/`, but each module is a five-line shim that
re-exports `<name>_impl` from `cli.py`; `cli.py` is still ~2200 lines and holds every command
implementation plus USFX parsing, Anki package readers, annotation, and cloze selection. That
is well past the 800-line ceiling in AGENTS.md and makes every command change touch one file.

## Acceptance

- Command implementations live in their `commands/<name>.py` module, not as `*_impl` shims.
- Domain helpers are extracted into focused modules (USFX/corpus parsing, Anki package I/O, annotation, cloze selection).
- `cli.py` retains only the Typer app wiring and genuinely shared helpers, and is under the 800-line ceiling.
- CLI behavior and command registration are unchanged; the existing suite passes untouched apart from import paths.
- Done as a red/green refactor per the TDD requirement in AGENTS.md.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
