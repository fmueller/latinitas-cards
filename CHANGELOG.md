# Changelog

Notable user-facing changes to Latinitas Cards are documented here.

## [Unreleased]

### Fixed

- `generate --recipe` and `preview --recipe` keep existing claim reviews when filtering
  output recipes. Editing recipes in a saved profile still requires renewed reviews.
- Corpus phrase candidate reports omit exact word-token glosses, which remain available
  to lexical cards. Malformed phrase spans still appear as `invalid_span` entries.

### Added

- Profiles with `user` terminology accept English (`en`) language tags and render
  grammatical answer labels on principal-part, phrase and form-parsing cards in English.
- English profiles also render phrase prompts and answers, principal-part front labels
  and comparison text in English. Latin-first prompts and other tags are unchanged.
- Reviewed form translations and phrase references can be English. A phrase reference in
  a language other than the card's is withheld, even when accepted.
- `corpus-cards profile validate` checks a corpus-generation profile, which names the
  corpora, target language, recipes with budgets, reading, limits and ranking weights.
- `corpus-cards select` ranks reviewed corpus exercises, keeps earlier selections and
  fills free budgets; its candidate report explains scores, filters and exclusions.
- `corpus-cards deselect` records a reason and keeps exercises out of future selections;
  `--restore` returns them, requiring confirmation if their context has changed.
- `corpus-cards preview`, `export` and `inspect --check` show reviewed selected exercises,
  create an initial Anki CSV and detect stale, unreviewed or changed exported cards.
- `corpus-cards plan` hands reviewed corrections to managed CSV updates, preserves
  withheld notes and reports explicit deselections as blocked manual retirements.
- `corpus normalize --adapter tei` imports one Perseus/EpiDoc TEI edition with its own
  citations, header provenance and an optional `--work` ID. Notes, apparatus readings and
  `choice` alternatives stay with each passage, inline and editorial markup is kept as
  spans, gaps warn, and `--allow-unknown` keeps unknown elements in passage text in
  `source_extra`.
- `annotations import-translations` turns a translation corpus aligned by citation into a
  patch of passage translations to plan, approve and apply; it reports unmatched passages.

### Fixed

- Corpus commands report validation errors for deeply nested JSON instead of failing
  with an internal error.

### Limitations

- Switching a profile's language tag makes all of its reviewed claims stale; review them
  again in the new language.
- `preview --help` and `generate --help` still describe `--terminology user` as German.
- Corpora the TEI adapter writes are rejected by v0.3.0 releases.

## [0.3.0] - 2026-10-20

### Fixed

- `annotations plan` rejects output and output-lock hardlinks to nested default
  manifest locks as aliases, even while the normalizer holds the reservation.
- `annotations apply` rejects output and output-lock aliases to reserved writer
  locks, including symlinked default-manifest DATA locks, before creating files
  or reporting a held lock as busy.
- Corpus claim readers retain live accepted or withheld reviews regardless of where
  stale copied bindings appear, while still ignoring genuine duplicate live reviews.
- `corpus show`, `corpus validate` and `annotations inspect --claims` report
  duplicate review decisions instead of internal errors, leaving those claims open.
- `annotations segment` rejects outputs aliasing corpus, claims and default manifest
  locks, including symlinked default-manifest DATA locks, before writing or reporting
  contention; ordinary sidecar output remains unchanged.

### Added

- `corpus show`, `corpus validate` and `annotations inspect` accept `--claims`
  to join explicit annotation reviews and report stale decisions without writing inputs.
- PyPI installation and upgrade guidance for the planned v0.3.0 release, with optional annotation extras.
- `corpus config` reports effective versions, paths, digests and value origins
  without modifying inputs; setup now reports missing flags instead of prompting
  when no terminal is available.
- `annotations apply` atomically appends planned automatic and reviewed proposals,
  reports pending reviews, skips applied records, and requires re-planning when
  pending operations' decisions, dependents, conflicts or validity change.
- `annotations plan` classifies patches with whole-patch safety facts, and
  `annotations approve` records explicit operation reviews without changing sidecars.
- `policy validate FILE [--json]` checks versioned sidecar-write policies,
  rejecting unknown fields and automatic supersede/retract permissions.
- `annotations inspect` and `corpus validate --annotations` report outdated anchors;
  `corpus show` provides bounded passage views with only current, live annotations.
- `annotations segment` writes reproducible sentence/token sidecars, preserves
  existing proposals, and safely skips byte-identical reruns without a timestamp.
- `cloze` reads normalized JSONL corpora with source citations and optional
  language-selected translations, including alongside XML, text and CSV inputs.
- `corpus normalize --adapter usfx` preserves markup, notes and headings in citable
  JSONL, rejects unsafe/oversized/deep XML, and can retain unknown markup with a warning.
- `corpus normalize --adapter text` creates validated JSONL with stable reviewed
  passage refs or labeled citations, and safely resumes interrupted manifest writes.
- `corpus validate CORPUS [--json]` checks normalized corpus records, identities,
  ordering and canonical text without modifying the input, reporting all findings.

### Changed

- Legacy Vulgate commands reject XML with DTDs/entities, over 64 MiB of input,
  or element depth above 64, without writing output.

### Limitations

- Publication is maintainer-gated; annotation extras remain experimental, and
  corpus/sidecar formats may change before 1.0. No new native Anki evidence is claimed.

## [0.2.1] - 2026-10-14

### Fixed

- Principal-part answer context and plain Personal Notes remain readable in
  Desktop night mode with manually installed reference CSS v4, independently of panel appearance.
- `claims record` rejects structurally invalid authored phrase references before
  saving any batch decisions, preserving previous reviewed answers.
- `phrases export` and `claims inspect --check` reject unsafe generated field controls
  while preserving tab/CR/LF and valid literal text; CSV checks retain embedded CR bytes.
- `phrases` applies the configured palette and light/dark/auto appearance to both
  translation and recognition cards using the manually installed reference CSS v4.
- `claims inspect` preserves escaped literal angle-bracket prose and following
  context in generated-card text for workspace, CSV and supported package inspection.

### Added

- Morphology profiles accept `appearance: auto` with manually installed reference
  CSS v4; light/night toggles are verified in isolated Linux Anki Desktop collections.
- Principal-part answers show individually reviewed German form translations for
  both recipes, with explicit sense/provenance and no added front-side hints.
- Generated principal-part and form-parsing answers offer Latin (default) or German
  grammatical terminology, with preview overrides and explicit legacy profile upgrades.
- `phrases` inspects and saves reviewed deck vocabulary, sense, form and bounded
  pattern selections, then previews explicit translation/self-assessment or form-recognition
  recipes with stable exercise identities, separate combination/answer approvals and offline
  claim inspection, deterministic approved CSV export and managed retained-slot updates.
- `managed` supports separately approved answer/presentation updates on retained
  principal-part, parsing and phrase cards; creation and retirement still require manual resolution.

### Limitations

- Phrase generation is bounded to reviewed preposition–noun, noun–genitive,
  verb–noun and adjective–noun patterns; uncertain readings stay withheld.
- CSV installs no templates/CSS and omission never withdraws cards. Initial import,
  old-layout setup and blocked withdrawal need reviewed manual resolution.
- Native verification is synthetic Linux Desktop only; Mobile, macOS/Windows,
  sync and live collections remain unverified. Optional annotation has unresolved dependency advisories.

## [0.2.0] - 2026-10-08

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
- Principal-part previews count accepted reviewed and withheld assessed claims separately
  from roles missing claim assessments, with sample denominators and no accuracy claim.
- `managed` refuses decks built with older pre-release note layouts or IDs.

### Fixed

- `managed` rejects unsafe active controls such as NUL before CSV handoff and
  reports backup/freshness attestations and the manual import boundary explicitly.
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
- Experimental annotation extras still include urllib3 2.7.0 with two high and one
  moderate network advisories; avoid untrusted downloads or proxy-sensitive use.

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

[Unreleased]: https://github.com/fmueller/latinitas-cards/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/fmueller/latinitas-cards/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/fmueller/latinitas-cards/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/fmueller/latinitas-cards/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/fmueller/latinitas-cards/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/fmueller/latinitas-cards/releases/tag/v0.1.0
