---
id: T-009-prepare-v0-1-0-release-candidate
title: Prepare the v0.1.0 release candidate
status: todo
priority: medium
spec_ref: specs/v0.1.0.md#release-readiness
dependencies:
    - T-004-preview-and-deterministic-csv-export
    - T-011-validate-the-representative-deck-and-german
updated_at: "2026-09-22T16:09:47Z"
---

# T-009-prepare-v0-1-0-release-candidate Prepare the v0.1.0 release candidate

## Description

Harden the deck-first workflow and prepare an exact release candidate without presenting
existing experimental APKG mutation, corpus generation, or grammatical parsing as stable
release promises. Publication is a separate final task.

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
- `pyproject.toml`, release documentation, and the prepared tag version agree on `v0.1.0`.
- The candidate identifies the exact commit approved for the publication task.

## Verification Notes

- TODO: record verification evidence and the verify run timestamp.

## Implementation Notes

- Do not create the tag or GitHub release in this task.
