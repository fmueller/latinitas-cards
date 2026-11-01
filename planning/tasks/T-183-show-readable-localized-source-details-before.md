---
id: T-183-show-readable-localized-source-details-before
title: Show readable localized source details before technical provenance
status: completed
priority: high
spec_ref: specs/v0.4.0.md#tei-xml-adapter
dependencies:
    - T-182-separate-localized-citations-from-source-provenance
updated_at: "2026-11-01T00:59:14Z"
---

# T-183-show-readable-localized-source-details-before Show readable localized source details before technical provenance

## Description

Present readable German/English source details before the existing lossless JSON
in the T-182 Citation disclosure. This is a presentation-only follow-up under
the existing v0.4.0 TEI contract; no schema amendment is needed.

## Acceptance

- Outer disclosure shows localized labeled source file, work, edition, passage,
  supplied URL/revision, source-file digest, licence and existing limitations.
- Complete unchanged canonical provenance JSON is inside a second localized
  Technical details disclosure. Both disclosures default closed.
- Preserve verified passage links, escaping, absence diagnostics, CSV layout,
  identities, selections and all provenance meaning. Supplied metadata does not
  imply rights clearance or identity with a live resolver's source revision.
- Test both languages, hostile/missing metadata and lossless CSV recovery.
  Inspect native Anki light/dark states and pointer/keyboard interactions using
  actual pinned Caesar BG 1.1.1 and Aeneid 1.11 in an isolated synthetic profile.
- Keep T-173 blocked and byte-untouched. No release or publication actions.

## Verification Notes

- TDD: both German/English CSV regression cases failed on the old renderer's
  missing readable fields. After implementation all 47 focused note/output
  tests passed, retaining hostile metadata, control escapes and selection checks.
- Dedicated simplifier found no useful changes. Independent conventional review
  approved the low-risk presentation diff with no findings; final formatter-only
  test edit was independently verified to preserve the reviewed semantics.
- Final exact-candidate Ruff -> mypy -> pytest chain passed: 3232 passed,
  seven skipped, one existing serializer warning, 491.31 seconds. Full format
  check passed for 476 files. Spec and T-173 bytes remain unchanged.
- Official Anki 26.09.3 acceptance passed on independently hash-verified product
  tree `c54166a5bb878d95437dee8841aedc694d07d3d7` (planning at base):
  all 24 reading/language/disclosure/light-dark states inspected. Native dark
  surroundings and fixed-light answer panels retain their separate palettes.
- Native audit: 16 notes/cards, all nine localized fields, unchanged canonical
  JSON/labels/hrefs, other 23 exported fields exact, IDs/GUIDs/scheduling unchanged,
  zero revlog. Native quote-entity normalization is compared semantically, not
  misreported as post-editor byte equality. CSS trims only its trailing newline.
- Genuine pointer plus Space/Enter/Tab interactions passed in light and dark,
  including neighboring controls. Clips decode fully; sequential frame/contact
  sheets were inspected because direct video analysis was unavailable. Exact
  keyboard actions rely on the native owner's exercised-input record, not still
  images alone. Final quiescent database audit retained the same receipt hash.
- Native receipt SHA256:
  `f6ca54152014bdea7820c99a6bcee3be36b3721883b11c5134e070a880df9fe2`.
  Dark/light clip SHA256 respectively:
  `12e2dead892a0764374c808c6c9549e236bd7ffd7f619bcd72f8fe711b602411`,
  `05c36a9a3bcea6b52257de022bf71d21c8730299f30349a27b84b68cf1215ddd`.

## Implementation Notes

- Only Citation presentation, localized labels and definition-list CSS change.
  Canonical JSON creation, resolver allowlist, diagnostics and all persistence,
  identity and selection contracts remain unchanged.
- Retained final authenticated candidate archive SHA256:
  `d6388ac8c4291de857df027e196de87eea66086d6d539c5354b6fe81b8fc2383`.
- 2026-11-01T00:59:14Z: verification pass
- 2026-11-01T00:59:14Z: Presentation-only localized source details and nested lossless technical provenance accepted; T-173 remains blocked and unchanged.
