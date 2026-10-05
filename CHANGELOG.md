# Changelog

Notable user-facing changes to Latinitas Cards are documented here.

## [Unreleased]

### Added

- `form-parsing` previews individually reviewed contextual Latin features and exports
  eligible exercises for a separate manual note type; automatic acceptance stays disabled.
- `managed plan` exposes deterministic destination-aware JSON reviews;
  `managed approve` binds selected compatible operations to their exact import footprint.
- `managed emit` writes approved content/tag CSV handoffs with backup records;
  `managed observe` and `managed reconcile` record offline results before retrying.

### Changed

- Managed destination reviews reject historical identities and incompatible layouts;
  explicit fresh-start plans require backup and a separate destination with new schedules.
- Principal-part answers offer reviewed German explanations and restrained muted or
  monochrome light/dark comparisons; static is the default, with manual CSS setup.
  Fourth-form cards still require explicit PPP/supine review.
- Principal-part profiles can retain reviewed pipe alternatives and the bounded
  `poet.` hint as source evidence; unresolved targets are withheld in both recipes.
- Principal-part previews separate wholly skipped entries from generated entries
  with omission/review warnings, and exports preserve extraction evidence.

### Fixed

- Managed CLI JSON escapes unsafe Unicode terminal controls in successful reviews
  and reports while retaining exact content when decoded.

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
