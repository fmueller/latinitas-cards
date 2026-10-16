---
id: T-110-preserve-supported-usfx-markup-and-notes-and
title: Preserve supported USFX markup and notes and reject unsafe or unknown XML
status: completed
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies:
    - T-109-normalize-usfx-sources-into-the-corpus-model-with
updated_at: "2026-10-16T07:48:27Z"
---

# T-110-preserve-supported-usfx-markup-and-notes-and Preserve supported USFX markup and notes and reject unsafe or unknown XML

## Description

Extend the strict USFX reader of T-109 with the remaining supported markup, notes
and headings, `--allow-unknown`, and full XML hardening, per
`specs/v0.3.0.md#usfx-normalization` (first paragraph and the synthetic-fixture Acceptance
sentence) and the passage fields `notes`, `headings`, `source_markup`, `source_extra` of
Fixed Contracts **Corpus records**. Today the strict reader accepts only `usfx`, `book`, `h`,
`c`, `v`, `ve` and rejects a DOCTYPE; the legacy ElementTree projection is unaffected here
(T-112 applies the hardening to it). The full-corpus oracle is T-111.

## Acceptance

- `p` and `q` are transparent containers (their text is passage text; no record);
  `wj` text stays in `text` and is recorded in `source_markup` as `{"element": "wj", "span":
  [start, end], "attributes": {}}` with code-point offsets into the canonical `text`; `<w>`
  text stays in `text` and its attributes (e.g. `s`, `l`, `x-morph`) are kept verbatim as
  strings in a `source_markup` entry with element `w` and span. Spans are checked by slicing.
- Notes `f`, `x`, `fe` and headings `s` leave `text`: each becomes one entry in passage
  `notes` (`{"kind": "f"|"x"|"fe", "text": ...}`, canonicalized, descendants flattened in
  document order and never checked as unknown) or `headings` (`{"text": ...}`); a note
  attaches to the verse it occurs in, a heading attaches to the next passage. A note outside
  any verse (for example between `<ve/>` and the next `<v/>`) attaches to the next passage,
  like a heading; with no next passage in the work it exits 3 `invalid_structure`. Nested
  `<f><fr>1:1</fr><ft>...</ft></f>` flattens without an `unknown_element` failure.
- An unknown element fails with exit 3 (`unknown_element`, one diagnostic listing names
  and counts, e.g. `zz=2, foo=1`) and writes nothing; with `--allow-unknown` the run exits 0,
  keeps the element's text in `text`, records `{"element", "attributes", "span"}` in the
  passage's `source_extra`, and reports the same names and counts as a `severity: "warning"`
  diagnostic (stderr line in text mode).
- Hardening, each exit 3 with nothing written: a DOCTYPE (`unsafe_xml`), an internal or
  external entity declaration (`unsafe_xml`, via `EntityDeclHandler`), an external entity
  reference (`unsafe_xml`, `SetParamEntityParsing(XML_PARAM_ENTITY_PARSING_NEVER)` and a
  rejecting `ExternalEntityRefHandler`), input over 64 MiB (`input_too_large`, checked from
  `stat` and while streaming, without reading the whole file), element depth over 64
  (`depth_exceeded`). The over-size test uses a sparse or generated tmp file, not a
  committed one; the depth fixture has depth 65 and a control with depth 64 passes.
- Synthetic fixtures in `tests/fixtures/corpus/usfx/markup/` cover milestone and container
  books and verses, `c@n`, `p`, `q`, `wj`, `w` attributes, nested `f`, `x`, `fe`, `s`, a `3a`
  verse, a duplicate verse (exit 3), an unknown element (failure, then `--allow-unknown`),
  non-NFC input, a DTD and an entity declaration; each has a committed expected JSONL or
  expected exit/code pair, and every success output passes `corpus validate`.
- The billion-laughs fixture (entity declarations) fails fast: the test asserts exit 3
  within a 5 s subprocess timeout and that no entity text appears in any output.
- The JSON Schema for passages (`docs/schemas/latinitas-corpus.v1.schema.json`) constrains
  `notes`, `headings`, `source_markup` and `source_extra` item shapes; the drift test passes.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

### Initial workflow-v3 pause, 2026-10-16 (historical; superseded below)

- Understand: fetched and verified HEAD and origin/main at
  `d615e93f7690dd33079a7c6a179e6887a631431a`; clean worktree. Full task, USFX
  section and Fixed Contracts read. Active spec v0.3.0 SHA256 remained
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
  `taskrail validate` and `next --json` selected this exact task with no owner
  conflict before start, follow-up creation and block; commands used mise exec.
- TDD: new markup/security tests initially produced 10 failed, 1 passed.
  Supported markup was rejected, size/depth diagnostics absent, and pending
  metadata failed as unknown. Corrected one malformed test XML and hand-counted
  endpoints before production changes. Literal fixtures assert canonical slices,
  notes and heading passage IDs independently of generated snapshots.
- Offset TDD: two non-NFC reordered-combining-mark cases failed with swapped
  spans (2 failed, 2 controls passed), then passed with NFD-origin matching.
  All successes validate through the public corpus command. Synthetic snapshots
  were generated with `regen-corpus-fixtures.py --target markup`; tests never
  regenerate them. Legacy goldens, source corpus and provenance are unchanged.
- Initial checks: exact ruff/mypy/pytest chain passed (139 typed files,
  1,421 tests, 76.18 s). Initial failures for diagnostic documentation and old
  free-form source_extra test assumptions were corrected, not suppressed.
  The old float object now correctly yields float_value plus invalid_record.
- Dedicated simplifier Task loaded code-simplifier and added the no-markup
  canonical_text fast path only. Parent inspected it and reran 136 focused
  tests successfully. No Unicode simplification was accepted.
- Separate parallel read-only code-reviewer Tasks: General (general reviewer),
  Security (security reviewer plus security-review companion), Python (Python
  reviewer plus python-patterns), and FastAPI/Pydantic (schema portions only).
  Each loaded its index/reviewer references. Database omitted: no SQL, database
  schema, migrations or locking changed. Other framework/domain lanes do not
  apply. Three specialist lanes stay within the default budget.
- Fresh candidate-validation Task loaded code-reviewer, validated S-1 and T110-1,
  and deduplicated G-1 into S-1. No rejected candidates. Python returned
  "No concrete task-relevant findings."

### Initial verbatim findings and dispositions (S-2 reassessed below)

- G-1 (General, medium): "Markup offset mapping can exhaust memory on valid
  input near the 64 MiB limit." Duplicate of S-1; same fix and evidence below.
- S-1 (Security, high): "Marked USFX input can amplify the allowed 64 MiB source
  into multi-gigabyte memory use during span canonicalization, allowing a crafted
  input to exhaust memory and terminate the process." Evidence: the original
  per-code-point arrays at usfx_adapter.py:252-256,298; review measured 405,936
  KiB peak RSS for a 2 MiB marked verse. Fixed through strict TDD: subprocess
  regression failed at 295,632,896 bytes of RSS growth against a 64 MiB ceiling;
  sparse boundary offsets, streamed origins and packed combining-run indices
  reduced growth to 21,610,496 bytes (91,568 to 112,672 KiB). Independent hash,
  text length and empty span remained correct. No new input restriction.
- T110-1 (Pydantic, medium): "Corpus validation accepts markup spans that extend
  beyond the passage text, despite spans being defined as offsets into canonical
  text." Evidence: corpus_model.py:232-233 and corpus_validation.py:141-151;
  text abc with span [4,5] validated. Fixed through strict TDD: public validation
  tests for both fields initially accepted out-of-bounds spans (2 failed), while
  both valid end-boundary controls passed. Field-specific bounded_spans now
  rejects them as invalid_record without misclassifying passage identity.
- Disposition tests: 5 passed after both fixes. Additional scratch validation:
  5,000 seeded Unicode/whitespace interval differential probes passed; no scratch
  test or helper was retained.
- Final code gates: `uv run ruff check` passed; `uv run mypy` passed for 139
  source files; `uv run pytest -v` passed 1,429 tests in 78.37 s. `mise run check`
  exited 0, including format, the same mandatory checks, commit/push/author and
  mutation-guard suites. One expected Pydantic warning comes from the test that
  deliberately corrupts a typed model to exercise serializer float rejection.
- Real CLI manual check: unknown fixture exits 3 and creates no output; explicit
  allow-unknown exits 0 with aggregate warning on stderr; corpus validate exits
  0. Security controls cover stat and streaming size checks, exactly 64 MiB,
  depth 64/65 inside markup and opaque notes, DTD/internal/external declarations,
  billion-laughs under a 5 s subprocess timeout and no entity payload output.
- Fresh disposition-verification Task loaded code-reviewer, General and Security
  references, independently passed 152 focused tests, and marked S-1 and T110-1
  resolved. It raised S-2; a separate fresh candidate validator confirmed it:
  "A small USFX input containing many empty markup elements can still amplify
  memory use substantially, despite the bounded span-mapping change."
  Evidence: usfx_adapter.py:399-401,313-352 retains one model per required entry;
  reviewer measured about 47.6 MB growth for 50,000 empty w elements. Parent
  reproduction: 200,053 source bytes, 50,000 retained entries, text a,
  51,961,856 bytes RSS growth, 138,776 KiB peak. S-2 remains unresolved, not
  silently deferred as safe. Compact/spooled retention crosses this task's size
  guard; arbitrary count limits would need contract review.

### Initial stop and handoff (historical)

- One review/fix/recheck cycle completed; new S-2 candidate validated separately.
  Stopped at the explicit size guard rather than widen into the corpus pipeline.
  Created T-128-bound-dense-usfx-markup-retention-without-dropping through the
  supported CLI at 2026-10-16T00:50:52Z, without changing existing dependencies
  or priorities; blocked T-110 through the CLI at 2026-10-16T00:50:59Z.
- No taskrail verify/complete, commit or push: the security review gate is not
  satisfied. Changes remain local and uncommitted; remote main is not a delivery
  of this work. Hooks installed and maintainer identity checked before delivery
  was considered. No spec, release/tag/deployment, legacy projection, data,
  dependencies or T-038 gate changes. T-127 remains its existing separate task.
- Observed local next selector after block: T-113-define-annotation-sidecar-records-and-the.
  This cycle does not execute that task or the new follow-up. The source owner
  must decide the broader retention work and integration of this blocked work.

### Same-cycle resumption and corrected S-2 disposition

- Resumed the same medium-mode T-110 cycle and checkout on user direction;
  no replacement worker or second task implementation. Fetched origin/main;
  accepted base and pinned active-spec SHA256 still match the values above.
- A fresh independent code-reviewer candidate-validation Task read the full
  task, USFX acceptance, Fixed Contracts and actual retention code, loaded
  General/Security/Python guidance, and independently passed 67 focused tests.
  Its conclusion: "REJECTED as a current must-fix contract violation; retain
  the measured memory amplification as future hardening."
- Original S-2 finding remains verbatim above. Its measured overhead is real:
  the 200,053-byte / 50,000-entry input added 51,961,856 RSS bytes and completed
  successfully. The original failure/impact claim, "A crafted input can exhaust
  memory before reaching the 64 MiB source limit, terminating normalization,"
  was a potential risk, not an observed OOM or crash. No explicit total-memory
  budget or markup-count ceiling was identified in pinned v0.3.0. Byte/depth
  rejection, entity hardening and required metadata retention pass their actual
  acceptance criteria. Reject the current-blocker classification, not the data.
- Final S-2 disposition: future hardening, not fixed, no active-contract change.
  User assigned broader planning to
  https://ampcode.com/threads/T-01a143a7-979b-72d9-95b6-d8d6cd94b6c2,
  drafting the future v0.8.0 `bounded-memory-dense-usfx-markup-retention`
  subheading under Performance And Accessibility Hardening. No count cap or
  spooling implementation was added to T-110.
- Reconciled T-128 through supported `task repoint --spec-ref` at
  2026-10-16T07:43:29Z to the already published local heading
  `specs/v0.8.0.md#performance-and-accessibility-hardening`. Taskrail v0.4.0
  exposes no cancellation command. This supersedes its initial active-v0.3.0
  blocker role while preserving ID/evidence. The planning thread owns its
  refinement/repoint to the new subheading after delivery; no duplicate ID,
  off-spec start, status hand-edit, priority or dependency edit occurred.
- Supported `unblock` returned T-110 to todo at 2026-10-16T07:43:30Z with this
  rationale; validate/next selected this exact task again before `start`.
  Historical block notes are preserved, not silently erased. S-1 and T110-1
  remain fixed with their original red/green and independent verification.
- Fresh final disposition-verification Task loaded code-reviewer with General,
  Security/security-review, Python/python-patterns and Pydantic guidance. It
  independently passed 211 focused tests, verified G-1/S-1 and T110-1 resolved,
  and accepted S-2 as explicit future hardening, not fixed, with no demonstrated
  current-contract failure. Its conclusion: "No concrete task-relevant findings."
  It could not invoke plain taskrail on its PATH; parent independently ran
  `mise exec -- taskrail validate` and `next --json` successfully. No historical
  benchmark or lifecycle check was invented as newly rerun by that reviewer.
- Final rerun after reconciliation: `uv run ruff check` passed; `uv run mypy`
  passed for 139 files; `uv run pytest -v` passed 1,429 tests in 91.16 s.
  `mise run check` exited 0, including format, 1,429 tests in 87.02 s, and all
  commit-message, push-message, author and mutation-guard suites. The single
  expected Pydantic negative-test warning is unchanged. Product code has not
  changed since the preceding full review/fix cycle.
- Re-fetched before finalization: origin/main and local HEAD still equal the
  accepted base; active-spec hash is unchanged. Hooks reinstalled and maintainer
  author identity confirmed. Exactly T-110 proceeds to verify/complete; T-128
  remains off-spec todo for the future planning thread, and T-038 stays blocked.

## Implementation Notes

- Extend the expat event reader in `src/latinitas_cards/usfx_adapter.py`; keep markup
  span bookkeeping in raw-offset space, then map spans through the whitespace
  canonicalization (build an offset map while collapsing whitespace) so spans index the
  final `text`.
- Hardening uses only stdlib expat handlers; add no dependency. Document the mechanism in
  `docs/license-compatibility-audit.md` or the USFX section of `docs/normalized-corpus.md`
  in one line.
- If the pinned `corpus_model` passage models need the item shapes above, change only
  those optional fields; `passage_hash` still covers only `text`.
- Expected JSONL files are snapshots written once by `scripts/regen-corpus-fixtures.py
  --target markup` (T-104); tests never regenerate them. In addition, each markup fixture
  test asserts at least one literal `source_markup` span (checked by slicing to a literal
  string), one literal note `{kind, text}` and one literal heading attachment `passage_id`,
  written in the test before the implementation.
- Decision: with `--allow-unknown` the unknown element's text stays in `text` so
  `passage_hash` covers it; `source_extra` records element, attributes and span. Unknown
  character-style wrappers would otherwise drop verse text; do not "fix" this.
- Size guard: changes in `usfx_adapter.py`, the passage model fields and fixtures. File a
  follow-up rather than widen.
- 2026-10-16T00:50:59Z: Validated security finding S-2: dense required markup still amplifies memory (50000 empty w elements, about 200 KB input, about 47.6 MB RSS growth). Compact or spooled retention exceeds T-110 size guard; remediation tracked in T-128-bound-dense-usfx-markup-retention-without-dropping. No count cap or spec change authorized; review gate unresolved.
- 2026-10-16T07:43:30Z: Fresh independent contract-focused validation rejects S-2 as a current must-fix: measured dense-markup overhead completed successfully; no OOM/crash or specified total-memory budget breach demonstrated. Broader retention is future hardening in T-128, repointed via CLI to existing v0.8.0 performance-and-accessibility-hardening; planning thread owns refinement. S-2 is not fixed; no caps or spooling added.
- 2026-10-16T07:48:17Z: verification pass
- 2026-10-16T07:48:27Z: Verification passed at 2026-10-16T07:48:17Z after exact mandatory chain and mise check, independent simplification, separate reviewer lanes, TDD dispositions and fresh final disposition verification. S-1/G-1 and T110-1 resolved. S-2 is measured future-hardening overhead, not a demonstrated current-contract failure or fixed issue; T128 repointed via supported CLI to v0.8.0 and left unstarted for the planning owner. No current spec, cap, spooling or legacy changes.
