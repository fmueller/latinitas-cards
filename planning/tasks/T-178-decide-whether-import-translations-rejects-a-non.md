---
id: T-178-decide-whether-import-translations-rejects-a-non
title: Decide whether import-translations rejects a non-Latin CORPUS
status: completed
priority: medium
spec_ref: specs/v0.4.0.md#context-and-translation-recipes
dependencies: []
updated_at: "2026-10-30T22:28:48Z"
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

- Maintainer decision recorded before behavior change: require the `CORPUS`
  header language's primary subtag to be `la`, checked case-insensitively from
  metadata only. Reject other labels with `unsupported_language`, exit 3.
  The narrow Parallel translations Fixed Contract clarification is authorized;
  existing `PARALLEL` validation and reviewed-claim eligibility stay unchanged.
- 2026-10-29T18:27:33Z: maintainer decision: reject a non-Latin CORPUS in import-translations
- 2026-10-30T22:07:37Z: Maintainer approved: CORPUS must declare primary language la; reject other labels with unsupported_language, exit 3; inspect metadata only and preserve PARALLEL checks and reviewed-claim eligibility.
- 2026-10-30T22:28:48Z: verification pass
- 2026-10-30T22:28:48Z: Approved maintainer decision implemented: CORPUS primary metadata language must be la, otherwise unsupported_language exit 3. Existing PARALLEL checks, writer protections and reviewed eligibility preserved. Independent simplification/review/adversarial/disposition verification and integrated offline QA passed; mandatory exact full validation chain passed. Remote future-spec planning reconciled without changing semantic candidate. Parent owns T172 acceptance/advancement; no successor started.
