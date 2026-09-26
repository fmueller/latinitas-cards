---
id: T-028-inherit-parent-anki-note-tags
title: Inherit parent Anki note tags on generated cards
status: todo
priority: high
spec_ref: specs/v0.1.0.md#principal-part-card-generation
dependencies: []
updated_at: "2026-09-26T15:42:31Z"
---

# T-028-inherit-parent-anki-note-tags Inherit parent Anki note tags on generated cards

## Description

Every generated completion or recognition note must inherit all tags from its
source Anki note, in addition to configured generated-note tags. Anki tags belong
to notes; cards created from those notes share them. Currently `sources.py`
does not read `notes.tags`, canonical records have no tag metadata, and
`generation.py` passes only `profile.tags` to both recipes.

## Acceptance

- APKG and COLPKG adapters retain source-note tags as metadata distinct from
  ordinary fields. Tags are associated with the correct parent, never inferred
  from neighboring records or unioned across the source deck.
- Every exercise generated from a parent inherits all of that parent's valid
  tags for both recipes. Profile/default/CLI-configured generated tags are
  additive and do not replace inherited tags.
- Combine inherited and configured tags deterministically without duplicates;
  preserve hierarchical tags and valid Unicode names. Define normalization
  consistently with native Anki semantics and retain existing output-safety
  checks. Invalid tags produce actionable diagnostics, not silent tag loss.
- Untagged sources retain the existing configured-tag behavior. Do not guess
  that an arbitrary CSV field is Anki tag metadata or silently change CSV
  profile mappings; explicitly document the supported source-tag boundary.
- Preview and exported Anki Tags metadata contain the same combined tags.
  Changing tag membership/order never changes source or Latinitas identities;
  repeated generation remains byte-deterministic and never mutates the source.
- Synthetic APKG/COLPKG regressions include two parents with disjoint tags,
  overlapping configured tags, duplicate tags, hierarchy/Unicode, and an untagged
  parent. Check all generated descendants and the serialized Tags column.
- Native first/repeat imports verify inherited tags on resulting notes, stable
  identity, and no duplicates. Document native tag merge behavior separately
  from source inheritance; do not claim automatic removal of obsolete tags or
  reconciliation of user-added destination tags without testing it.

## Verification Notes

- Source inspection on 2026-09-26 confirmed that the Anki adapter selects only
  `id, mid, flds, guid` and generation supplies only `profile.tags`.
- Use synthetic tagged fixtures; do not commit the supplied private deck or
  its tags. Record the mandatory Ruff/mypy/pytest chain and native evidence
  when implemented.

## Implementation Notes

- Keep tag metadata outside semantic identity inputs. Reuse canonical adapters,
  generation, and existing tag validation rather than adding a separate path.
