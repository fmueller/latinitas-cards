---
id: T-016-refresh-user-facing-documentation-for-v0-1-0
title: Refresh user-facing documentation for v0.1.0
status: todo
priority: medium
spec_ref: specs/v0.1.0.md#release-readiness
dependencies:
    - T-004-preview-and-deterministic-csv-export
updated_at: "2026-09-22T19:04:42Z"
---

# T-016-refresh-user-facing-documentation-for-v0-1-0 Refresh user-facing documentation for v0.1.0

## Description

Rewrite the README as a concise, user-facing entry point for Latinitas Cards and audit the
repository's documentation before the v0.1.0 release. Follow common open source documentation
conventions, keep setup and first-use guidance easy to find, and clearly distinguish the stable
deck-first release workflow from experimental commands.

## Acceptance

- The README gives a concise overview of the project, its v0.1.0 status and supported workflow,
  prerequisites, installation, quick-start usage, and links to detailed documentation.
- The README follows common open source conventions by making development and contribution
  guidance, support expectations, and license information easy to find without duplicating
  detailed reference material.
- User-facing examples cover profile creation, preview, generation, first import, and repeat
  import, and are checked against the released CLI commands and options.
- Stable v0.1.0 behavior is clearly separated from legacy or experimental APKG mutation, corpus,
  annotation, and parsing capabilities; no documentation implies that out-of-scope features are
  complete or supported release promises.
- All tracked documentation, including root-level documents and `docs/`, is reviewed for stale,
  contradictory, duplicated, or broken content and links; identified issues are corrected or
  explicitly recorded for follow-up before release.
- The final README is scannable and concise, directing readers to focused documents rather than
  becoming a comprehensive command reference.
- Documented setup and representative commands are smoke-tested, and the repository's required
  validation checks pass.

## Verification Notes

- TODO: record verification evidence and the verify run timestamp.

## Implementation Notes

- Preserve required maintainer and contributor guidance while removing user-facing duplication.
- Keep the install guidance added by T-017 through T-019: Python 3.13 or 3.14, the CPU-only
  `annotate` extra, and the highly experimental `annotate-gpu` extra.
