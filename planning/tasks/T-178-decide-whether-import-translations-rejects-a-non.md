---
id: T-178-decide-whether-import-translations-rejects-a-non
title: Decide whether import-translations rejects a non-Latin CORPUS
status: blocked
priority: medium
spec_ref: specs/v0.4.0.md#context-and-translation-recipes
dependencies: []
updated_at: "2026-10-29T18:27:33Z"
---

# T-178-decide-whether-import-translations-rejects-a-non Decide whether import-translations rejects a non-Latin CORPUS

## Description

Resolve whether `annotations import-translations` must reject a non-Latin
`CORPUS`. The v0.4.0 Context And Translation Recipes section calls it the Latin
corpus, but the Parallel translations Fixed Contract defines language rejection
only for `PARALLEL`. T-161 therefore does not add an unspecified rejection.
The current command accepts a German corpus with a German parallel corpus and
`--language de`; a maintainer must decide whether to change that contract.

## Acceptance

- Record the maintainer's decision about the `CORPUS` language requirement.
- If rejection is required, define its diagnostic and exit code before changing
  behavior, then cover it with a regression test.
- Preserve the existing `PARALLEL` language checks and reviewed-claim eligibility.

## Verification Notes

- T-161's independent review confirmed that only `PARALLEL` language rejection is
  specified. This decision is not needed to implement remaining v0.4.0 tasks.

## Implementation Notes

- 2026-10-29T18:27:33Z: maintainer decision: reject a non-Latin CORPUS in import-translations
