---
id: T-105-validate-normalized-corpora-with-corpus-validate
title: Validate normalized corpora with corpus validate
status: completed
priority: high
spec_ref: specs/v0.3.0.md#normalized-jsonl-corpus-model
dependencies:
    - T-103-define-the-cli-result-envelope-exit-codes
    - T-104-define-versioned-corpus-records-with-a-canonical
updated_at: "2026-10-15T22:52:02Z"
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
- Encoded-ref fixtures accept uppercase `%HH` escapes and literal separator dots;
  malformed escapes (`%`, `%2`, `%GG`, `%2e`) and raw non-ASCII refs are exit 1
  `invalid_record` findings, not header-gate errors. Preserve encoded spelling verbatim;
  `%252E` and `%2E` are distinct valid refs, not duplicate refs after decoding.
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

### Task outcome

| Status | Outcome | User/Product Impact |
| --- | --- | --- |
| Implemented and reviewed | Read-only `corpus validate CORPUS [--json]` | Ordered actionable findings and record counts, with structured exits |

### What changed

| File/Area | Change | Why |
| --- | --- | --- |
| `corpus_validation.py` | Pure streaming record validator plus path/sidecar seam | Semantic checks without changing T-104 reader/models |
| `commands/corpus.py`, CLI registration | New group with only `validate` | Text summary/diagnostics and JSON envelope |
| `cli_contract.py`, contract docs/schema/matrix | Registered codes and optional text renderer | Reuse T-103 boundary, preserve existing output defaults |
| Corpus tests, CLI ownership test, release metadata test | Public behavior regression coverage; allow pending changelog entries after the released section | Preserve legacy behavior while adding the command |
| `CHANGELOG.md` | Pending user-facing capability | No release/tag/publication performed |

### Workflow evidence

| Step | Evidence | Result |
| --- | --- | --- |
| 1 | Fetched origin; HEAD and origin/main both `81711956fa28d79b7909d70e01c46c20dd7f5c88`; clean tree; validate/next selected T-105 before start | No conflicting ownership or scope drift |
| 1 | Read full task, root guidance, normalized corpus section and Fixed Contracts; spec v0.3.0 SHA256 `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704` | Pinned spec unchanged; no show/config/sidecar behavior or schema-parity expansion |
| 2 | `uv run pytest tests/unit/corpus_validation_test.py -q` before registration | 29 failed for absent command (exit 2); then 29 passed |
| 2 | `uv run pytest tests/unit/corpus_validation_test.py -k invalid_work -q` | 2 failed/2 passed: unhashable malformed work caused internal 70; guard fixed it |
| 3 | Focused corpus/contract tests | 60 passed |
| 3 | Initial mandatory chain | 2 failed/1228 passed: command-ownership test needed the new group; old release assertion forbade any pending changelog. Corrected expectations, full chain 1230 passed |
| 4 | Dedicated Task loaded personal `code-simplifier` | No changes retained; rejected unbound loop closure simplification (Ruff B023); full checks and 100 focused tests passed |
| 5 | Separate parallel General, Python, Security Tasks loaded personal `code-reviewer` and indexed references | General ECC code-reviewer; Python reviewer + python-patterns; Security reviewer + security-review/common rules |
| 5 | Lane selection | General mandatory; Python for changed implementation; Security for untrusted JSONL/output. No DB, web framework, UI, ML, healthcare or network configuration boundary changed |
| 5 | Fresh candidate-validation Task loaded `code-reviewer`, Python/Security references | General candidate validated; Python candidate validated and deduplicated; Security candidate validated; float/schema omission independently reproduced |
| 6 | `uv run pytest tests/unit/corpus_validation_test.py -k 'same_code or float_does or lone_surrogate' -q` before fixes | 9 failed, 1 passed, 33 deselected; missing findings, missing envelope/internal failure, or unsafe accepted Unicode |
| 6 | Same regression cases after fixes; full corpus test file | 43 passed. First green attempt found JSON-mode key conversion and exception-name mypy issue; corrected within this fix cycle |
| 7 | Fresh exact `uv run ruff check && uv run mypy && uv run pytest -v` | Ruff passed; mypy 128 files clean; 1240 passed |
| 7 | `mise run check` | Ruff/format/mypy, 1240 pytest cases, all commit/author/push/mutation harness checks passed |
| 7 | Fresh disposition-verification Task loaded `code-reviewer`, General/Python/Security references and companions | Independently reran mandatory chain: 1240 passed; corpus 43 passed; all F1/F2/F3 resolved; "No concrete task-relevant findings." |
| 7 | Review/fix/recheck count | One cycle, none deferred or unresolved |
| 8 | Pre-lifecycle refetch, selector, validation and spec hash at 2026-10-15T22:51:10Z | Same base/spec, selector still owned active T-105; Taskrail state valid |

### Review findings and dispositions

- F1 (General T105-1), verbatim: "The validator drops distinct same-code defects
  from a record, contrary to the all-findings diagnostic contract."
  Fixed: retain distinct schema and semantic diagnostics with field locations.
  Regression combines two citation defects and unknown work plus wrong order;
  asserts four ordered diagnostic messages. Red/green evidence above.
- Python candidate FINDING 1, verbatim: "When one record has multiple independent
  defects classified as invalid_record, validation emits only the first diagnostic
  for that line, contrary to the requirement to report all findings."
  Validated, deduplicated with F1 (same code-suppression root cause); fixed by F1.
  No rejected candidates.
- F2 (Security S-001), verbatim: "Reject or safely handle unpaired Unicode
  surrogates from corpus input before using them in hashes or diagnostics;
  otherwise malformed-but-valid JSON can prevent the promised validation response."
  Fixed: detect lone surrogates recursively, emit `invalid_record`, escape unsafe
  diagnostic strings and skip hashing invalid text. Preserve dictionary keys with
  Python-mode model dumps. JSON/text tests cover refs, text, notes and metadata keys.
- F3 (candidate validator), verbatim: "A float finding can prevent the validator
  from reporting independent schema defects on the same record, leaving the
  all-findings result incomplete."
  Fixed: revalidate raw float-bearing records with the accepted adapter, keeping
  `float_value` plus independent errors but omitting the duplicate float invariant.
  Regression combines `source_extra` float and invalid citation scheme.

### Manual verification and limits

- `uv run latinitas-cards corpus validate tests/fixtures/corpus/valid/usfx-bcv.jsonl --json`:
  exit 0, empty diagnostics, counts header=1/corpus=1/work=2/passage=3.
- Invalid hash text invocation: exit 1, stdout `1 errors, 0 warnings`, stderr
  `error hash_mismatch test-corpus:p000001: Line 4: Hash differs from canonical text`.
- `corpus --help` lists only `validate`. Input bytes and mtimes asserted unchanged;
  whole-file read APIs forbidden in streaming test. T-102 golden tests passed unchanged.
- Hooks installed with `mise exec -- lefthook install`; maintainer identity inspected.
- No follow-up work identified. Only this task was implemented. T-038 remains blocked.
- Final lifecycle verification timestamps are recorded by Taskrail below on completion.

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
- 2026-10-15T22:51:54Z: verification pass
- 2026-10-15T22:52:02Z: Verified pass at 2026-10-15T22:51:54Z after strict TDD, dedicated simplification, separate General/Python/Security reviews, candidate validation and fresh disposition verification. All three findings fixed; no deferrals or follow-ups. Mandatory chain and mise run check pass (1240 tests); T-102 goldens unchanged.
