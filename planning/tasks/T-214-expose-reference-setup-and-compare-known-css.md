---
id: T-214-expose-reference-setup-and-compare-known-css
title: Expose reference setup and compare known CSS versions
status: todo
priority: medium
spec_ref: specs/v0.5.1.md#reference-setup-and-css-diagnostics
dependencies:
    - T-213-retain-checkpoints-and-valid-reviews-across
updated_at: "2026-11-05T22:11:24Z"
---

# T-214-expose-reference-setup-and-compare-known-css Expose reference setup and compare known CSS versions

## Description

Deliver FR-6 reference-setup CLI and source-collection CSS diagnostics from the
canonical generator, including nominal layout versions. No direct Anki writes.

## Acceptance

- reference-setup --format md|json [--out PATH] exposes ordered fields, named front/
  back templates, full CSS and actual layout/CSS versions. Output alone permits
  manual setup and agrees with canonical docs/templates.
- inspect/preview require exact fields/templates before reporting known-old versus
  current CSS. Distinguish custom/unknown CSS and nonmatching templates; CSV-only
  comparison reports unavailable.
- Never imply inspection of an unseen destination or replace custom CSS. Empty-model
  APKG is not required and must not be promised without native evidence.

## Verification Notes

Test canonical stdout/file formats and current/known-old/custom/nonmatching fixtures.
T-216 owns Linux Desktop setup/import. Run mandatory ruff → mypy → pytest chain;
follow full spec compatibility/evidence/permission requirements.

## Implementation Notes
