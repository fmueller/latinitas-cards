---
id: T-043-reject-lone-surrogates-in-authored-input
title: Reject lone surrogates in authored input validation
status: todo
priority: high
spec_ref: specs/v0.1.1.md#authored-note-import-format
dependencies: []
updated_at: "2026-10-04T13:15:17Z"
---

# T-043-reject-lone-surrogates-in-authored-input Reject lone surrogates in authored input validation

## Description

Fix the medium-severity Unicode validation gap found during synthetic,
adversarial v0.1.1 testing. A valid UTF-8 JSONL file can encode a lone surrogate
in a JSON string. The authored loader currently accepts it, validate/preview
report success, and export fails during UTF-8 serialization without identifying
the source line or field. Reject non-UTF-8-encodable string values at the import
validation boundary while retaining whole-file diagnostic recovery.

Evidence: https://ampcode.com/threads/T-01a106d3-a5dc-7716-b644-a1b50079ec6d

## Acceptance

- Strict red/green regression reproduces a first-row meaning of `\ud800`
  followed by a valid row. Loader diagnostics identify the first line and field,
  retain the later valid row, and never expose an exportable valid result.
- Lone high and low surrogates are rejected throughout authored string fields,
  including optional content, provenance, keys, language and tags, without
  tracebacks or unprintable diagnostics. Valid Unicode, including non-BMP
  characters and correctly paired JSON surrogate escapes, remains accepted.
- Real authored validate, preview and export all exit nonzero for such input,
  including skipped or filtered-out invalid rows; report aggregate line/field
  errors rather than a late serialization failure.
- Invalid export creates no output and leaves existing outputs and input bytes
  unchanged. Tests cover recovery and multiple invalid rows as appropriate.
- Keep schema and opaque provenance contracts intact; do not normalize or
  silently replace invalid strings or add unrelated input restrictions.
- Complete workflow-v3 review and disposition verification; `uv run ruff check`,
  `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Record decisive regression results and the Taskrail verification timestamp;
  do not commit references to gitignored artifact paths.

## Implementation Notes
