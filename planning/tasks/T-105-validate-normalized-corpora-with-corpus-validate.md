---
id: T-105-validate-normalized-corpora-with-corpus-validate
title: Validate normalized corpora with corpus validate
status: todo
priority: high
spec_ref: specs/v0.3.0.md#normalized-jsonl-corpus-model
dependencies:
    - T-103-define-the-cli-result-envelope-exit-codes
    - T-104-define-versioned-corpus-records-with-a-canonical
updated_at: "2026-10-14T05:42:59Z"
---

# T-105-validate-normalized-corpora-with-corpus-validate Validate normalized corpora with corpus validate

## Description

Add the `corpus` command group with its first read-only command, `corpus validate CORPUS
[--json]`, implementing the corpus checks of `specs/v0.3.0.md#normalized-jsonl-corpus-model`
("Acceptance: `corpus validate --json` accepts committed golden corpora ...") under the Fixed
Contracts **Versioned documents**, **Identity and hashing** and **Exit codes** items. It uses
the T-104 reader/models and the envelope, exit-code constants and diagnostic registry from
T-103. `--annotations` (stale/orphaned/unmatched reporting) is added by T-114 and `--claims`
by T-125; `corpus show` (T-114) and `corpus config` (T-120) belong to other tasks.

## Acceptance

- `latinitas-cards corpus validate tests/fixtures/corpus/valid/{usfx-bcv,ordinal,wsl}.jsonl
  --json` each exit 0 with one `latinitas-cli-result` v1 document (`command: "corpus
  validate"`, `ok: true`, empty `diagnostics`, `result` with record counts per type).
- Exit 3 (gate failures: input rejected, nothing else is checked) with one registered code
  each for: a missing file (`unreadable`), a line that is not a JSON object (`invalid_json`),
  a first line that is not a header (`missing_header`), `format` other than
  `latinitas-corpus` (`unsupported_format`), and `version: 2` or `version: 1.0`
  (`unsupported_version`). A non-JSON line later in the file is also exit 3
  (`invalid_json`), after reporting findings for earlier lines.
- Exit 1 with distinct codes, each from a committed one-defect fixture in
  `tests/fixtures/corpus/invalid/`: malformed `corpus_id` (`invalid_corpus_id`), duplicate
  `native_ref` (`duplicate_native_ref`), non-NFC `text` (`non_nfc_text`), uncollapsed
  whitespace (`non_canonical_text`), `passage_hash` mismatch (`hash_mismatch`), stale `norm`
  (`norm_mismatch`), `passage_id` not `corpus_id:native_ref` (`passage_id_mismatch`), a
  non-string citation value (`invalid_citation`), a float anywhere (`float_value`), and
  schema/order violations such as a missing `corpus` record or a gap in `order`
  (`invalid_record`). Each test asserts the exact code set, not just the exit code.
- A file with several defects reports all of them in file order (no fail-fast after the
  header gate); a file with both a gate failure and findings exits 3 (precedence 3 > 1).
- Text mode (default) prints one line per diagnostic (`<severity> <code> <ref>: <message>`)
  to stderr and a summary line (`<n> errors, <m> warnings`) to stdout, through
  `run_contract`, with the same exit code; `--json` writes nothing but the
  envelope to stdout. Diagnostics carry `ref` = `passage_id` (or `native_ref` when the ID is
  unusable), `path` = the CORPUS argument as given, `item` = null.
- Every code above is in the T-103 registry and the `corpus validate` rows of
  `docs/agent-cli-contract.md`, and the registry-equals-docs test passes. Validation is
  read-only: file bytes and mtimes of CORPUS are unchanged after every case.
- Validation streams the file line by line: a test monkeypatches `Path.read_text` and
  `Path.read_bytes` to raise and validation of `valid/usfx-bcv.jsonl` still exits 0.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- New `src/latinitas_cards/corpus_validation.py` (pure: iterator of records in,
  diagnostics out) and `src/latinitas_cards/commands/corpus.py` (Typer sub-app `corpus`,
  registered like `commands/claims.py`; no logic in `cli.py` beyond registration).
- Do not declare CORPUS with Typer `exists=True` (Click would exit 2 without an envelope);
  open it in the command and map failures to exit 3 `unreadable`.
- Header gate first: the reader's typed errors map to exit 3; everything after a valid
  header maps to exit 1 findings, except a later non-JSON line (exit 3).
- Register `invalid_corpus` (exit class 3) for consumers that reject an invalid corpus as
  input; `corpus validate` itself reports the finding codes.
- Invalid fixtures are derived from `valid/ordinal.jsonl` by a single targeted edit each,
  generated once by `scripts/regen-corpus-fixtures.py --target invalid` (T-104) and
  committed with a short `tests/fixtures/corpus/invalid/cases.json` mapping file ->
  expected exit and codes.
- Use T-103's `run_contract` for `--json`; never print Rich markup to stdout in JSON mode;
  warnings go to diagnostics with `severity: "warning"`.
- Leave a seam (`validate_corpus(path, *, annotations=None, claims=None)`) so T-114 and
  T-125 can add sidecar checks without reshaping this command.
- Size guard: one domain module plus the new command group. File a follow-up rather than
  widen.
