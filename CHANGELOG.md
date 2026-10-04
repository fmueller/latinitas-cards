# Changelog

Notable user-facing changes to Latinitas Cards are documented here.

## [Unreleased]

### Added

- Validate and preview authored JSONL notes without changing files, with whole-file
  diagnostics, duplicate merging, and filters by kind, section, reference, and tag.
- Export selected authored vocabulary, form, and QA notes as repeatable Anki CSVs;
  edits retain note identity and Personal Notes stays unmapped on re-import.
- Extract Markdown study notes with the repository agent skill; review stable keys,
  preserved skip decisions, and new/changed/missing items in a validated preview.

### Fixed

- Authored validate, preview, and export diagnostics display terminal controls as
  visible escapes instead of executing controls embedded in invalid input.

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

[Unreleased]: https://github.com/fmueller/latinitas-cards/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fmueller/latinitas-cards/releases/tag/v0.1.0
