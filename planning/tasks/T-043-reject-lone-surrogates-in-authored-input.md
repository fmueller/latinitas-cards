---
id: T-043-reject-lone-surrogates-in-authored-input
title: Reject lone surrogates in authored input validation
status: completed
priority: high
spec_ref: specs/v0.1.1.md#authored-note-import-format
dependencies: []
updated_at: "2026-10-04T13:46:59Z"
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
- Taskrail verification passed at 2026-10-04T13:46:09Z, after all review and
  disposition verification. Final chain: `uv run ruff check` passed;
  `uv run mypy` passed (65 source files); `uv run pytest -v` passed (557 tests).
- Strict TDD: `uv run pytest tests/unit/authored_import_test.py -k
  'lone_surrogates or non_bmp' -q` reported 32 failed, 2 passed before the
  schema change; 34 passed afterward. High and low surrogate cases cover all
  content fields, optional content, provenance, key, language, and tags with
  later-row recovery. Raw non-BMP and paired JSON escapes retain exact values.
- Targeted import/export/preview tests passed (100); dedicated simplifier loaded
  `code-simplifier`, made no edits, and ran all authored tests (126 passed).
- Separate independent General, Python, and Security lanes loaded
  `code-reviewer` and their ECC guidance (Python patterns and security review
  companions). Each concluded verbatim: "No concrete task-relevant findings."
  Security covers input trust-boundary and data-loss risk; Python covers typed
  schema behavior. Database/framework/domain-specialist lanes were omitted:
  no database, web framework, or corpus/ML behavior changed.
- Fresh candidate validation concluded: "No candidate IDs were supplied, so
  there were no candidates to validate or deduplicate. No concrete missed task
  issues found." No findings required fixes or deferrals. Fresh disposition
  verification concluded: "No unresolved or newly introduced task-local issues
  found; acceptance criteria are met." One review cycle was sufficient.
- Real synthetic CLI subprocess checks: 12 surrogate-negative validate,
  preview, and export invocations exited 1; six raw/paired-escape Unicode
  positive invocations exited 0 and retained non-BMP text in UTF-8 CSV.
  Original first-row meaning plus later valid row reported
  `line 1, vocab.meaning: Value error, must be valid UTF-8 text without lone
  surrogates` and `Invalid: 1 errors; diagnostic matches: 1; not exportable`.
  Aggregate skipped/filtered cases also reported line 2 dictionary_form and
  `Invalid: 2 errors`, never a late export serialization error or traceback.
- Exact byte comparisons preserved input and absent/existing output snapshots,
  including an existing CSV sentinel containing NUL and non-UTF-8 bytes.
  Input SHA-256 values were
  `a5765be5431179881956ba49eda81951c52d49717234d1fe51fba6bae4487c51` and
  `af369fbfea8b7e9d86e9c6c6a82d077430ffe162e8a31712e73d378f60c16a61`.

## Implementation Notes

- An annotated UTF-8 string validator is shared by schema string fields;
  literals retain existing enum validation. Validation returns accepted text
  unchanged and uses an ASCII-only failure message, preserving opaque
  provenance and T-042 terminal diagnostic escaping. Selection/export code
  needed no changes. Advisory invalid-kind behavior remains out of scope.
- 2026-10-04T13:46:09Z: verification pass
