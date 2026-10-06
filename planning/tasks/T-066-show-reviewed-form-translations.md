---
id: T-066-show-reviewed-form-translations
title: Show reviewed translations of principal-part forms
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#reviewed-form-translations
dependencies:
    - T-068-publish-v0-2-0
updated_at: "2026-10-06T19:55:00Z"
---

# T-066-show-reviewed-form-translations Show reviewed translations of principal-part forms

## Description

Render a reviewed user-language translation of each present principal part on the
answer side (tested form and four-role comparison), e.g. `amāvī` → *ich habe geliebt*,
following `specs/v0.2.1.md#reviewed-form-translations`. Requested by the maintainer
during T-062 native acceptance. Decisions should flow through the T-064 CLI review route.

## Acceptance

- Translations are reviewable claims per form, role and selected sense; only accepted
  ones render, on the answer side only, for both principal-part recipes.
- Candidates may come from the meaning field, rules or optional analyzers but never
  bypass review; absent/withheld roles show no translation.
- Fixtures cover a regular verb, a deponent, a multi-sense entry, and PPP versus supine.
- Translations follow the user language regardless of the terminology setting; note/card
  identities and slots stay unchanged.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
