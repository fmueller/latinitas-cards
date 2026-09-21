---
id: T-009-harden-and-release-v0-1-0
title: Harden and release v0.1.0
status: todo
priority: medium
spec_ref: specs/v0.1.0.md#release-readiness
dependencies:
    - T-001-assisted-deck-profiles
    - T-002-stable-generated-note-identity
    - T-003-principal-part-card-generation
    - T-004-preview-and-deterministic-csv-export
updated_at: "2026-09-21T22:38:31Z"
---

# T-009-harden-and-release-v0-1-0 Harden and release v0.1.0

## Description

Harden the deck-first workflow and cut the first release without presenting existing
experimental APKG mutation, corpus generation, or grammatical parsing as stable release
promises.

## Acceptance

- A sanitized realistic fixture exercises the complete assisted-profile through
  repeatable-CSV workflow.
- README and CLI documentation cover profile creation, preview, generation, first import,
  repeat import, and the experimental status of legacy mutation/corpus commands.
- `THIRD_PARTY_NOTICES.md` and the dependency-license audit match the current uv lock,
  including CLTK and its runtime dependency graph.
- `CHANGELOG.md` contains a v0.1.0 entry describing the stable deck-first scope and explicit
  exclusions.
- The exact release commit passes ruff, strict mypy, and the full pytest suite on every
  supported Python version.
- The `v0.1.0` tag matches `pyproject.toml` and a GitHub release exists for that tag.

## Verification Notes

- TODO: record verification evidence and the verify run timestamp.

## Implementation Notes

- Release publication happens only after all dependencies are completed and verified.
