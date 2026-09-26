---
id: T-026-render-source-html-safely
title: Render source HTML as safe readable study content
status: todo
priority: high
spec_ref: specs/v0.1.0.md#principal-part-card-generation
dependencies: []
updated_at: "2026-09-26T15:42:31Z"
---

# T-026-render-source-html-safely Render source HTML as safe readable study content

## Description

Native Anki Desktop testing of the unreleased v0.1.0 candidate showed literal
`<div>...</div>` text in generated German meanings for both recipes. Source HTML
is escaped directly rather than converted to readable content. Correct this
presentation defect without trusting or executing source markup.

## Acceptance

- Completion and recognition display readable source meanings without literal
  formatting tags or encoded formatting entities. Preserve meaningful block and
  line boundaries, German characters, Latin macrons, and ordinary plain text.
- Audit other source-owned displayed text for the same issue and handle it at
  the appropriate normalization boundary; do not change principal-part role
  semantics, approved omission behavior, or identity derivation.
- Script, event-handler, and other active source content cannot execute. Do not
  fix the issue by rendering arbitrary source HTML unescaped.
- Synthetic regressions cover nested formatting, div/br boundaries, entities,
  plain text, and hostile markup in both recipes and exported CSV.
- Inspect native Anki rendering of both recipes after the fix, and verify
  deterministic output, unchanged logical IDs, and source immutability.

## Verification Notes

- Origin: native Anki Desktop 26.9.3 testing on 2026-09-26. Reproduce with
  synthetic text such as `erste Bedeutung<div>zweite Bedeutung</div>`; do not
  commit private source decks, source-derived counts, or identifiers.
- Record the required Ruff, mypy, and pytest chain and native rendering outcome
  when implemented. No implementation verification is claimed by this task.

## Implementation Notes

- Inspect `generation.py` meaning extraction/rendering and existing text
  normalization helpers before selecting the smallest safe change.
