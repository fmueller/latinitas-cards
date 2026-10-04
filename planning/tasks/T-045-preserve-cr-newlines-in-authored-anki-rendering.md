---
id: T-045-preserve-cr-newlines-in-authored-anki-rendering
title: Preserve CR newlines in authored Anki rendering
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-kinds
dependencies: []
updated_at: "2026-10-04T18:57:12Z"
---

# T-045-preserve-cr-newlines-in-authored-anki-rendering Preserve CR newlines in authored Anki rendering

## Description

Fix the medium native-Anki fidelity defect found in the third synthetic
whole-spec retest. Accepted CR-only content `left\rright` and opaque reference
`part\rtwo` survive CSV serialization but become `leftright` and `parttwo`
after native import and collection reopen for vocab, form and QA. Rendering
currently converts only LF to HTML breaks, despite documented newline handling.

Evidence: https://ampcode.com/threads/T-01a107a0-c767-736c-b08b-5903cf53afd9

## Acceptance

- Strict red/green regression demonstrates CR-only content and reference loss.
  At the HTML rendering boundary, CRLF, lone CR and LF each render as one
  `<br>` per logical newline, including mixed and consecutive line endings.
- Preserve source values and exact opaque-reference filters; do not normalize
  import strings, keys or identities as part of the rendering correction.
- Cover required and optional content and provenance across all three kinds,
  with HTML escaping preserved and no duplicate break for CRLF.
- Execute supported native-Anki first import and edited re-import, followed
  by close/reopen, for each kind. Newline boundaries survive as intended HTML,
  notes/cards/IDs remain stable, and nonempty Personal Notes remains untouched.
- Repeat real CLI export and verify deterministic UTF-8 CSV, correct mapping,
  exact filtering and unchanged input; existing terminal/Unicode fixes remain
  intact. Render representative affected cards and inspect their appearance.
- Keep scope to CR newline fidelity. NUL interoperability and unknown-kind
  advisories are not silently converted into unrelated behavior changes.
- Complete workflow-v3 review and disposition verification; `uv run ruff check`,
  `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Record decisive regression/native results and the Taskrail verification
  timestamp; do not commit references to gitignored artifact paths.

## Implementation Notes

- 2026-10-04T18:56:55Z: verification pass
- 2026-10-04T18:57:12Z: Verified 2026-10-04T18:56:55Z after workflow-v3: strict renderer/native RED to GREEN; 572 tests, ruff and mypy pass; native Anki 26.9.3 first/edited imports and reopen preserve logical newlines, identities/cards and Personal Notes for all kinds; repeated real CLI exports preserve source and exact CR reference filters. Browser-inspected first QA and edited all-kind answers. Dedicated simplifier plus General/Python/Security/Database review, fresh candidate validation and disposition verification complete; DB-1 rejected because native answers include FrontSide, no validated findings. NUL and unknown-kind advisories unchanged.
