---
id: T-177-decide-where-allow-unknown-keeps-tei-elements
title: Decide where --allow-unknown keeps TEI elements outside passage text
status: blocked
priority: medium
spec_ref: specs/v0.4.0.md#tei-xml-adapter
dependencies:
    - T-159-preserve-tei-notes-apparatus-and-editorial-markup
updated_at: "2026-10-29T13:37:03Z"
---

# T-177-decide-where-allow-unknown-keeps-tei-elements Decide where --allow-unknown keeps TEI elements outside passage text

## Description

Follow-up found during T-159 review. Fixed Contracts **TEI subset** (Text and markup)
says any other element fails `unknown_element` "unless `--allow-unknown` keeps it in
`source_extra`". `source_extra` entries are spans in canonical passage text, so T-159
keeps an allowed unknown element only where it stands in passage text. Elsewhere it has
no text position: between textparts or verse lines, inside a flattened note, heading or
variant (`del`, `rdg`, `sic`/`abbr`/`orig` in `choice`), as a child of `app` or `choice`
(for example `rdgGrp` or `witDetail`), or in an omitted empty passage. T-159 fails these
closed with `invalid_structure`, except that an unknown element in an omitted passage
is dropped like that passage's other text spans; `docs/normalized-corpus.md` documents
it. For the same reason `app`, `lem`, `rdg`, `choice` and `gap` inside flattened text fail
`invalid_structure`, although the contract's `gap` rule ("records an empty span and
warns `gap_in_text`") does not limit where a gap stands.

Keeping these would need a contract choice the spec does not make: for example an empty
`source_extra` span at the attachment point, a separate record field, or an explicit
failure rule. That changes the Fixed Contracts, so it needs a maintainer decision.

## Acceptance

- The maintainer records whether these positions fail closed (and the spec says so) or
  how `source_extra` (and `gap_in_text`) represent them.
- If the representation changes, the maintainer amends the spec and the TEI adapter and
  `tests/unit/corpus_tei_test.py` follow it.

## Verification Notes

- Not started.

## Implementation Notes

- 2026-10-29T13:37:03Z: maintainer decision: where --allow-unknown keeps unknown TEI elements without a canonical-text position (between textparts or lines, inside flattened notes, headings and variants, inside app or choice, or in omitted passages), which the Fixed Contracts TEI subset says are kept in source_extra but T-159 fails closed
