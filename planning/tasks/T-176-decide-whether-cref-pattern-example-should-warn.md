---
id: T-176-decide-whether-cref-pattern-example-should-warn
title: Decide whether the Aeneid line, Book cRefPattern example should warn
status: blocked
priority: medium
spec_ref: specs/v0.4.0.md#tei-xml-adapter
dependencies:
    - T-158-normalize-perseus-epidoc-tei-structure-citations
updated_at: "2026-10-29T12:30:37Z"
---

# T-176-decide-whether-cref-pattern-example-should-warn Decide whether the Aeneid line, Book cRefPattern example should warn

## Description

Follow-up found during T-158 review. Fixed Contracts **TEI subset** (Citation) says
the command warns `citation_mismatch` when the `refsDecl[@n="CTS"]/cRefPattern/@n`
values, read deepest first, differ case-insensitively from the unit path. The TEI XML
Adapter section and the T-158 acceptance give the Aeneid's `line, Book` as an example
that warns. The Perseus Aeneid uses textpart `@subtype="book"` and lines, so its unit
path is `book, line`; its patterns `line, Book`, read deepest first, are `Book, line`,
which equals the unit path case-insensitively and does not warn under the rule.

T-158 implements the binding rule (`tei_adapter.py`, pinned by
`test_cref_pattern_mismatch_only_warns`: `LINE, Book` does not warn, `Book, line`
warns). Making the Aeneid example warn would mean a case-sensitive comparison or a
different reading order, which changes the Fixed Contracts, so it needs a maintainer
decision: either the example is corrected, or the comparison changes.

## Acceptance

- The maintainer records which of the rule or the example is authoritative.
- If the rule changes, the spec is amended by the maintainer and the TEI adapter and
  its citation test follow it; otherwise the example text is corrected.

## Verification Notes

- Not started.

## Implementation Notes

- Fixed Contracts in `specs/v0.4.0.md` are binding; agents do not amend them.
- 2026-10-29T12:30:37Z: maintainer decision: whether the Aeneid line, Book cRefPattern example should warn, which the Fixed Contracts TEI subset deepest-first case-insensitive citation_mismatch rule currently treats as matching
