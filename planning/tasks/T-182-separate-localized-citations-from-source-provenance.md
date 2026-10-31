---
id: T-182-separate-localized-citations-from-source-provenance
title: Separate localized reader-facing citations from complete source provenance
status: completed
priority: high
spec_ref: specs/v0.4.0.md#tei-xml-adapter
dependencies: []
updated_at: "2026-10-31T18:13:07Z"
---

# T-182-separate-localized-citations-from-source-provenance Separate localized reader-facing citations from complete source provenance

## Description

Record the user's v0.4.0 requirement: "I want to ensure that v0.4.0 is creating
fully tracable references for provenance but also nice, reader-facing labels.
Of course, for both supported languages, German and English."

Separate a readable, localized author/work/passage citation from complete,
unambiguous source provenance. Anchor normalization and metadata work to
`specs/v0.4.0.md#tei-xml-adapter`; preserve the exercise/selection identity
contracts in `#corpus-first-profiles-and-exercise-identity` and the language
contract in `#german-and-english-output`. This is open requirements/engineering
work, not implementation permission or a spec amendment. Propose and review any
needed schema or Fixed Contracts changes before implementing them.

Current source evidence at candidate
`041989c54f32ed3a0ebc0481fd81776f2c368396`:

- `src/latinitas_cards/corpus_notes.py:190` formats `Citation` from
  `passage.citation.display` plus the corpus prefix of `passage.passage_id`.
  The retained Aeneid record for `1.11` has display `1.11`, passage ID
  `phi0690-phi003:1.11`, and book/line citation parts. That expression yields
  `1.11 (phi0690-phi003)`, not an author/work label. This is a source/record
  observation, not a fresh native screenshot assertion.
- The pinned Aeneid XML has `body/@xml:base` equal to
  `urn:cts:latinLit:phi0690.phi003.perseus-lat2`; combined with passage `1.11`,
  the edition-specific reference is
  `urn:cts:latinLit:phi0690.phi003.perseus-lat2:1.11`.
- `tei_adapter.py` currently maps edition/publisher/URL/licence/title metadata
  and derives a work ID through `_work_id`. The normalized header also retains
  raw source filename, byte count and SHA256. `corpus_notes.py` emits separate
  Source ID/Scope/Kind/Location fields. Inspect their full downstream contracts
  before diagnosing provenance loss: these observations do not prove that all
  provenance is retained, or that all provenance is lost at any boundary.

## Acceptance

- In both `de` and `en`, reader-facing citations show readable author, work and
  passage labels based on verified source metadata, with language-appropriate
  names/terminology and citation-unit presentation. Do not replace traceability
  with a friendly label or invent missing author/work information.
- Make the reader-facing reference clickable to the actual source passage when
  a reliable exact-passage URL can be verified. Investigate the actual supported
  Perseus resolver/deep-link format and verify work, edition and passage
  resolution for pinned Caesar BG `1.1.1` and Aeneid `1.11`. This task creation
  asserts no online resolver availability or verified URL; do not implement
  guessed URL patterns. Localize `de`/`en` labels without changing the accurate
  link target or full provenance.
- Distinguish an authoritative edition-specific passage link from a broader
  work/context fallback, and label any fallback honestly. For arbitrary TEI,
  missing metadata, offline use or no reliable resolver, provide a useful
  nonclickable citation with all available provenance and an explicit limitation;
  never fabricate CTS identities or emit guessed/dead links.
- Preserve safe, properly escaped HTML links through generation -> CSV -> native
  Anki import. Verify both the imported link target and actual clicking/opening
  behavior where the environment permits; explicitly record any unperformed
  browser-opening check. A clickable citation does not replace source
  URL/revision/digest provenance or silently change LatinitasID/schema semantics.
- Retain an unambiguous edition/work/passage identity independently of display
  wording. Preserve source URL, upstream revision and digest provenance wherever
  supplied; distinguish supplied metadata from derived references and local
  identifiers. Verify actual CTS bases/edition identifiers before constructing
  CTS passage references; never invent CTS identities for arbitrary TEI.
- Define explicit diagnostics and honest deterministic fallbacks when author,
  work, edition, URL, revision or other provenance metadata is absent. Missing
  values must remain distinguishable from verified metadata, not silently
  fabricated or relabelled as complete provenance.
- Prove provenance survives the supported normalize -> corpus generation ->
  CSV export -> fresh native Anki import path, not only the normalized source
  or reader-facing Citation field. Inspect actual exported fields and imported
  note content, including the relationship between the readable label and its
  complete source reference; record expected versus observed evidence.
- Exercise the actual pinned Perseus Caesar BG `1.1.1` (book/chapter/section)
  and Aeneid `1.11` (book/line) in both supported languages. Use upstream
  `PerseusDL/canonical-latinLit` revision
  `8b536bdf3ed1f43494055efe112433b4a58eab67`; record source digests and verify
  cited Latin/metadata independently, not with a synthetic substitute.
- Inspect LatinitasID, source identity, selection and context compatibility.
  A localized citation/metadata change must not silently change exercise IDs,
  duplicate notes, invalidate selections or drift existing provenance bindings.
  Document expected compatibility, and explicitly schema-review any proposed
  field/identity/selection migration before changes; preserve fixed identity
  vectors unless a separately approved contract change authorizes otherwise.
- Add focused regression coverage for both languages, the two citation unit
  structures, absent-metadata fallbacks, provenance handoff and compatibility.
  Native evidence must come from isolated synthetic profiles using these real
  source passages, without personal accounts, sync or real scheduled collections.
- Keep source and translation rights/provenance separate. Source identity or a
  friendly label is not permission to redistribute the corpus or translations.
- Implementation and release acceptance remain unperformed by this task's
  creation. No T-173 dependency, lifecycle change, unblocking or publication
  authorization is introduced.

## Verification Notes

- Implementation and bounded independent acceptance passed on product tree
  `28df85feffbf29fff2818bdae317e71e0e85a954` (planning state excluded).
- TDD: initial nine target failures passed after implementation; author ambiguity
  and review-found serialization/source-binding regressions were reproduced and
  fixed. Deliberate disposable regressions confirmed the localization, citation
  unit and warning assertions fail; restored checks passed.
- Dedicated simplification and high-risk conventional review completed. Five
  specialist lanes and four fresh adversarial lanes used one shared challenge
  round and one materially-new-evidence follow-up. All confirmed code findings
  were fixed and independently disposition-verified; no required finding remains.
- Mandatory Ruff -> mypy -> pytest chain passed: 3232 passed, seven skipped.
  Distribution build, sdist rebuild and artifact inspection passed. A final
  post-note chain precedes lifecycle completion.
- Independent public-CLI QA derived its oracle before reading implementation
  reports. Fresh pinned sources passed both languages through normalization,
  reviewed annotations/claims, selection, preview, export and inspection; IDs,
  sticky selections, hostile metadata, corrected text and old-reader behavior
  were checked. Six initial harness expectations were reconciled explicitly,
  without changing product code or acceptance requirements.
- Independent official native Anki 26.09.3 acceptance imported 16 fresh notes and
  16 cards. All 24 exported fields per note matched CSV byte-for-byte, Tags
  matched as sets and Personal Notes stayed empty. Four language/reading cases
  rendered collapsed/expanded provenance and opened the exact source passages.
  Actual Scaife edition, heading and Latin were inspected; online revision
  equality is explicitly not claimed. Native CSS lost only its final newline.
- Imported evidence hashes and four paired captures were checked locally. Both
  pointer clips decoded cleanly; decoded frames/contact sheets were inspected
  because direct video analysis was unavailable. Full source/native receipts
  remain retained, separate from the later parent-owned final demo.
- No release, tag, publication, deployment or T-173 lifecycle action occurred.

## Implementation Notes

- Original pinned XML SHA256: Caesar
  `15ec3ef2a36352742ddb75d4f7da23a143899b6b60ee3a7ca3192d4917ca1568`;
  Aeneid `e639e6a0e27427ef28031850c4d73aa2b63b9f2cf8a5cf584b37685c52fb874c`.
- The maintainer-approved spec amendment defines optional author, edition URN
  and supplied revision fields; absent optional fields preserve existing bytes.
  Enriched documents intentionally require the newer strict reader. Existing
  identities, context digests, selection rules, CSV columns and template slots
  are unchanged.
- The existing Citation field now carries localized verified-source labels and
  separate lossless source-provenance JSON. Exact links require the pinned source
  metadata, citation structure and current passage hash; corrected readings and
  unsupported sources remain nonclickable with explicit limitations. Source
  rights and translation provenance remain separate.
- 2026-10-31T18:13:07Z: verification pass
- 2026-10-31T18:13:07Z: Final post-note Ruff/mypy/pytest chain passed (3232 passed, 7 skipped). Reviewed frozen product tree and independent CLI/native acceptance passed; T-173 stays blocked, with no release authorization.
