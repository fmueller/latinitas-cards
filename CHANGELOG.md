# Changelog

Notable user-facing changes to Latinitas Cards are documented here.

## [Unreleased]

### Added

- `managed` updates notes you already imported without losing their review history:
  `capture` reads a closed backup, `adopt` records reviewed ownership, `plan` shows
  what would change, `approve` picks individual changes, `emit` writes an
  update CSV for Anki's import, and `observe`/`reconcile` record what Anki actually did.
- Principal-part answers can show individually reviewed German explanations and a
  comparison of the principal parts in a muted or monochrome style, light or dark.
  Choose it in the profile's `morphology` section and paste the reference CSS into Anki.
- `form-parsing` previews and exports reviewed parsing exercises (case, tense, and
  similar) for a separate note type; nothing is accepted automatically.

### Changed

- Profiles can opt in to keeping `|` alternatives and a trailing `poet.` hint from your
  source; forms that stay ambiguous are left off the cards instead of guessed.
- Previews list skipped entries separately from generated entries with warnings.
- `managed` refuses decks built with older pre-release note layouts or IDs.

### Fixed

- `managed plan` flags previously adopted notes missing from a complete destination
  as conflicts requiring review, rather than proposing to create them again.
- `setup --reconfigure` preserves hand-edited morphology settings; explicit morphology
  sections without a version now fail with instructions instead of assuming one.
- `managed approve` can select reviewed keep-destination decisions and tag-ownership
  changes even when the exported text and tags stay unchanged.
- Principal-part previews and answers distinguish linguistic review from unresolved
  source evidence, and count unreviewed fourth roles as generated-entry warnings.
- `managed` JSON output escapes hidden terminal control characters without changing
  the actual text.

### Limitations

- `managed` content and tag-only updates passed Anki Desktop 26.09.3's import dialog
  on test data with [specific settings](docs/managed-csv-desktop-verification.md), not
  arbitrary clients/collections. Skipped effects stay unresolved after observation.
- No command yet exports your current Anki state for `managed plan`; you build that
  JSON from a closed full backup ([guide](docs/destination-snapshots-and-file-transport.md)).
- `managed` cannot change card front/back text, so cards already in study don't get the
  new explanations or themes that way.
- Reviewing explanations and fourth-form (PPP vs. supine) labels currently needs the
  Python API; decks built only with the CLI leave both out.
- Explanation, theme, and `form-parsing` cards passed display checks only on Anki
  Desktop 26.09.2 and AnkiMobile 25.09 with test notes; the static comparison stays
  the default.

## [0.1.1] - 2026-10-04

### Added

- Validate and preview authored JSONL notes without changing files, with whole-file
  diagnostics, duplicate merging, and filters by kind, section, reference, and tag.
- Export selected authored vocabulary, form, and QA notes as repeatable Anki CSVs;
  edits retain note identity and Personal Notes stays unmapped on re-import.
- Extract Markdown study notes with the repository agent skill; review stable keys,
  preserved skip decisions, and new/changed/missing items in a validated preview.

### Fixed

- CSV exports preserve existing output and state files when temporary staging
  fails, including disk-full errors before all files have been staged.
- Authored Anki cards preserve CRLF, CR, and LF line breaks in content and
  citations, including after import and edited re-import.
- Authored validate, preview, and export diagnostics and successful export paths
  display terminal controls as visible escapes without changing file destinations.
- Authored imports reject lone Unicode surrogates with line and field diagnostics
  before selection or export, while preserving valid non-BMP Unicode.

### Limitations

- Import into Anki is manual; back up before updates and leave Personal Notes
  unmapped. Forced termination, power loss, and concurrent exports are not protected.
- Anki may normalize Unicode and remove NUL characters; authored grammar and
  translations need human review. This release is on GitHub, not PyPI.
- Experimental annotation extras lock urllib3 2.7.0, affected by two high and one
  moderate security advisories; the default authored-note workflow does not install it.

## [0.1.0] - 2026-10-02

First release: turn an existing Latin deck into German principal-part study cards
and export them for import into Anki, without modifying the source.

### Added

- Inspect CSV, APKG, and COLPKG sources and save reusable profiles with assisted
  field suggestions and explicit confirmation of uncertain mappings.
- Preview and generate principal-part completion and recognition cards, grouped
  under one note with shared personal notes and independently scheduled cards.
- Export repeatable Anki-import CSVs with stable note identities for matching on
  reimport; source files stay unchanged and `Personal Notes` stays unmapped.
- Review skipped or ambiguous entries and missing forms before export; withhold
  affected notes when regeneration would remove previously exported cards.
- Set up Anki using reference templates and first/repeat import guidance;
  pre-release note models require explicit review rather than silent conversion.

### Limitations

- Import into Anki is manual. Back up before updating: imports that change managed
  content can replace destination-only tags; there is no live collection merging.
- Legacy splitting/APKG mutation, annotation, Ollama analysis, and corpus cloze
  workflows remain experimental. This release is available on GitHub, not PyPI.

[Unreleased]: https://github.com/fmueller/latinitas-cards/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/fmueller/latinitas-cards/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/fmueller/latinitas-cards/releases/tag/v0.1.0
