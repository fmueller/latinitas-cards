---
id: T-029-model-coherent-learning-objects-with-stable-note
title: Model coherent learning objects with stable note identities
status: completed
priority: high
spec_ref: specs/v0.1.0.md#coherent-learning-objects-and-identities
dependencies:
    - T-002-stable-generated-note-identity
    - T-013-generate-principal-part-study-cards
    - T-036-preserve-csv-and-manifest-recovery-on-interruption
updated_at: "2026-09-27T11:25:23Z"
---

# T-029-model-coherent-learning-objects-with-stable-note Model coherent learning objects with stable note identities

## Description

Replace the unreleased recipe/role-derived note model with coherent learning objects.
Adapt identity.py, manifest.py, profile.py, notes.py, generation.py, and preview_export.py
rather than introducing a parallel stack. The linked spec supersedes the old contract.

## Acceptance

- Distinguish source note, lexeme/sense object, generated knowledge, profile, and card type.
  Confirm one coherent object or report unsupported/ambiguous multi-object sources.
- Derive note identity from stable source identity, reviewed object key, and fixed
  note-family namespace; exclude recipe/profile selection, mutable text, and versions.
  Keep card semantic keys separate and never merge distinct sources by matching words.
- Fix reproduced independent-manifest collisions: one-row amo and fero manifests both
  allocated csv-source-000001 and generated identical IDs. Persist immutable source scope
  for manifests and define scope for explicit CSV IDs, including future authored IDs.
  Paths, content and profiles are not scope. Moving/reordering/saving/reloading the same
  manifest retains IDs; independent manifests remain distinct even with the same local ID.
- Define bootstrap before deriving IDs: uninitialized read-only preview requests scope
  confirmation without writing state or minting apparently stable IDs. The first approved
  export commits a unique scope with assignments and output through T-036's recovery
  boundary, including for explicit CSV-ID sources. Missing established state requires
  recovery/review or explicit fresh start, not silent replacement. A copied manifest
  represents the same source; an independent source needs explicit new scope. Determinism
  includes persisted identity state, not identical source contents alone.
- Lock independently specified versioned identity input/output vectors, not expectations
  calculated with the production helper. Explicitly handle existing unscoped manifests
  through a compatibility/fresh-start decision; never silently reinterpret their IDs.
- Establish the authoritative note schema and ownership categories here: regular managed
  fields, user-owned Personal Notes and special transport metadata (Tags is not a regular
  note field). Derive CSV positions/directives from it. T-030 adds exact card fields/slots
  to the same contract; T-033 consumes and publishes it rather than redesigning it later.
- Export one deterministic CSV row per eligible object with LatinitasID first and source
  GUID/ID, location/deck, schema/generator and effective profile provenance.
- Synthetic fero and sum remain distinct; reordering, corrected gloss/spelling, and adding
  supported recipe knowledge retain established object IDs. Split/merge needs review.
- Preserve source fields, templates, and deck structure, safe HTML, parent tags, and
  Personal Notes exclusion from CSV. T-026/T-027/T-028 stay closed.
- Update old count/identity fixtures and docs explicitly without rewriting completed
  task history. No new lexical, mnemonic, or morphological recipe is required.

## Verification Notes

- Use red/green tests for grouping, identity, profile changes, provenance, immutability,
  row counts and independent external field/tag mapping. Cover bootstrap cancellation,
  failed export, read-only preview, existing-state retry and copied scopes; extend T-036's
  interruption regressions when changing identity-state persistence. Run the mandatory
  ruff/mypy/pytest chain after implementation.
- Record evidence when executed; this task is planning only. Native checks are in T-035.

## Implementation Notes
- T-036 establishes safe recovery before this task extends persisted identity state.
  Conditional template implementation follows in T-030.
- 2026-09-27T11:25:16Z: verification pass
