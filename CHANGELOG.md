# Changelog

All notable changes to Latinitas Cards are documented here. The project has not been
published yet; this entry describes the prepared release candidate.

## [0.1.0] - Unreleased (candidate)

Status: candidate awaiting user review. No tag or GitHub release has been created.

### Stable deck-first workflow

- Inspect an existing CSV, APKG, or COLPKG source and confirm an assisted, human-readable
  profile.
- Preserve stable source identities and derive immutable `LatinitasID` values for generated
  exercises.
- Generate explicitly selected German principal-part completion and recognition exercises
  with semantic role names, provenance, and a `de` language tag.
- Preview generated, skipped, and ambiguous entries before writing a deterministic UTF-8
  Anki text-import CSV with `LatinitasID` as the first column.
- Restrict generated CSV columns to managed fields: the note type keeps the user-owned
  `Personal Notes` field, but repeat imports never see a `Personal Notes` column to
  overwrite (legacy CSVs map it to Ignore on every import).
- Reuse the same logical identities when managed wording, HTML, glosses, or tags change;
  leave source inputs and the user-owned `Personal Notes` field untouched.
- Include a sanitized representative APKG fixture and an end-to-end assisted-profile,
  preview, export, and repeatability regression contract.

### Explicit exclusions

- No direct updates to a live Anki collection, scheduling verification, native Anki
  rendering verification, or automatic conflict/retirement application.
- No corpus-first generation, new corpus adapters, generalized USFX/cloze generation, or
  grammatical parsing cards in the stable workflow.
- No stable promise for legacy APKG mutation/split-note cloning, CLTK/Stanza annotation,
  optional Ollama disambiguation, or the legacy USFX `preview`/`generate` path; those remain
  experimental.
- No package publication, tag creation, or GitHub release; publication is the separate
  T-014 task after user approval.
