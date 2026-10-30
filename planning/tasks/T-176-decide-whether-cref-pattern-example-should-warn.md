---
id: T-176-decide-whether-cref-pattern-example-should-warn
title: Decide whether the Aeneid line, Book cRefPattern example should warn
status: completed
priority: medium
spec_ref: specs/v0.4.0.md#tei-xml-adapter
dependencies:
    - T-158-normalize-perseus-epidoc-tei-structure-citations
updated_at: "2026-10-30T21:01:40Z"
---

# T-176-decide-whether-cref-pattern-example-should-warn Decide whether the Aeneid line, Book cRefPattern example should warn

## Description

Follow-up found during T-158 review. Fixed Contracts **TEI subset** (Citation) says
the command warns `citation_mismatch` when the `refsDecl[@n="CTS"]/cRefPattern/@n`
values, read deepest first, differ case-insensitively from the unit path. The TEI XML
Adapter section and the T-158 acceptance originally gave the Aeneid's `line, Book`
as an example that warns. The Perseus Aeneid uses textpart `@subtype="book"` and lines,
so its unit path is `book, line`; its patterns `line, Book` in declaration order,
read deepest first, are `Book, line`. These match the unit path case-insensitively
and do not warn under the rule.

T-158 implements the binding rule (`tei_adapter.py`, pinned by
`test_cref_pattern_mismatch_only_warns`: `LINE, Book` does not warn, `Book, line`
warns, with both lists in declaration order). The approved maintainer decision keeps
the binding rule and corrects the examples; no runtime change is needed.

## Acceptance

- Retain the binding deepest-first, case-insensitive citation-label comparison.
- Correct the TEI XML Adapter and T-158 acceptance examples: `line, Book` in declaration
  order matches a `book, line` unit path without warning; `Book, line` in declaration
  order mismatches and warns `citation_mismatch` without changing the unit path.
- Leave runtime code and existing citation tests unchanged; verify the examples through
  those tests and independent public CLI probes for matching and mismatching labels.

## Verification Notes

- Documentation-only correction; no new regression test is needed because the existing
  parameterized citation test covers mixed-case matches, order mismatches and missing labels.

## Implementation Notes

- Fixed Contracts in `specs/v0.4.0.md` are binding; agents do not amend them.
- 2026-10-29T12:30:37Z: maintainer decision: whether the Aeneid line, Book cRefPattern example should warn, which the Fixed Contracts TEI subset deepest-first case-insensitive citation_mismatch rule currently treats as matching
- 2026-10-30T20:47:30Z: Approved maintainer decision: keep the binding deepest-first, case-insensitive cRefPattern comparison; correct the contradictory Aeneid example and affected acceptance prose without runtime changes.
- 2026-10-30T21:01:40Z: verification pass
- 2026-10-30T21:01:40Z: Approved decision retained the binding deepest-first, case-insensitive rule; corrected spec and T-158 examples. Independent simplification, review and exact-candidate CLI QA passed; full ruff/mypy/pytest chain passed before verify/complete.
