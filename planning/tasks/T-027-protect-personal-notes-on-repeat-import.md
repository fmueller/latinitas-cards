---
id: T-027-protect-personal-notes-on-repeat-import
title: Protect personal notes during repeat CSV imports
status: todo
priority: high
spec_ref: specs/v0.1.0.md#preview-and-deterministic-csv-export
dependencies: []
updated_at: "2026-09-26T15:42:31Z"
---

# T-027-protect-personal-notes-on-repeat-import Protect personal notes during repeat CSV imports

## Description

Native Anki Desktop 26.9.3 reopens the generated CSV with Personal Notes mapped
to its source column, even after the previous import used `(Nothing)`. Explicit
ignore mapping preserved the personal-note sentinel in the native test, but
relying on a remembered mapping is unsafe. Address this v0.1.0 export/import
safety gap without attempting to change Anki itself or introducing live writes.

## Acceptance

- Provide a safe documented export/import path that does not offer an empty
  generated Personal Notes value for accidental overwrite on repeated imports.
  Prefer omitting user-owned fields from import data while retaining them in
  the dedicated note type; verify the chosen approach in native Anki before
  settling the CSV contract. Do not rely solely on a warning or remembered map.
- Keep Personal Notes user-owned and available on generated notes. Update
  first-import schema instructions, repeat-import instructions, CSV headers,
  mapping tests, and CLI guidance consistently for the chosen solution.
- Explain that existing CSVs containing Personal Notes require `(Nothing)` /
  Ignore on every repeat import; Anki's previous selection may not persist.
- In a disposable native collection, import, add a nonempty personal note,
  review a card, change managed content, and repeat import through a freshly
  opened dialog. Verify preservation of personal content, scheduling, review
  history, card/deck placement, and stable IDs, with no duplicate notes.
- Both recipes, deterministic output, source immutability, and the documented
  note-type matching scope remain supported. No live collection, synchronization,
  direct-update engine, or automatic migration of user collections is required.

## Verification Notes

- Origin: native first/repeat/unchanged CSV imports on 2026-09-26. Mapping reset
  was observed; actual personal-note loss was not deliberately tested or claimed.
- Use synthetic sentinels and record automated regression checks plus fresh
  native import evidence. Do not commit private collections or source data.

## Implementation Notes

- Anki stores fields on notes, not separately on sibling cards. Preserve the
  existing dedicated note type's user-owned field even if it is absent from CSV.
