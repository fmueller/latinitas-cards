---
id: T-111-verify-full-clementine-vulgate-normalization
title: Verify full Clementine Vulgate normalization against an independent oracle
status: completed
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies:
    - T-110-preserve-supported-usfx-markup-and-notes-and
updated_at: "2026-10-16T08:11:54Z"
---

# T-111-verify-full-clementine-vulgate-normalization Verify full Clementine Vulgate normalization against an independent oracle

## Description

Add the single full-corpus test of `specs/v0.3.0.md#usfx-normalization` ("One module-scoped
test normalizes the full bundled file ...") so USFX normalization is checked against the real
Clementine Vulgate and an independent oracle, not only small fixtures. It exercises the
strict reader from T-109/T-110 on `data/lat-clementine.usfx.xml`
(4,652,378 bytes; elements `usfx`, `book`, `h`, `c`, `v`, `ve` only; milestone verses) and
pins its output digest. It also confirms the mutation sandbox copies `data/`. Legacy command
rewiring is T-112; no production behaviour is added unless the oracle exposes a
defect in the reader, which is then fixed here with a focused regression fixture.

## Acceptance

- `tests/unit/usfx_full_corpus_test.py` has one module-scoped fixture that normalizes the
  bundled file once (in-process call of the adapter's normalize function with
  `corpus_id="vulgata-clementina"` and `data/lat-clementine.descriptor.json`, writing to a
  `tmp_path_factory` file); all assertions reuse it and no other test normalizes the full file.
- It asserts source SHA-256
  `05bcb3f669d9b36deab6cf26950b8a24c7609b292e5af1abfa51717677701950` (before and after: source
  bytes never change), 73 `work` records, 1,334 distinct (work, chapter) citation pairs,
  35,809 `passage` records, zero duplicate `passage_id`s, and `order` = 1..35,809.
- Independent oracle: a regex scan over the raw source text (`re` on `<book id="...">`,
  `<c id="..."/>`, `<v id="..."/>`, tracking the current book and chapter; no import from
  `usfx_adapter`, `corpus_model` or `cli`) builds the expected `passage_id` set
  (`vulgata-clementina:BOOK.CH.V`) and the expected per-book verse counts; both equal the
  normalized output.
- Spot checks against literal expected strings: GEN 1:1 text and `passage_hash` equal the
  pinned vector from `tests/fixtures/corpus/schema/hash-vectors.json`, the first five
  passages equal `tests/fixtures/corpus/usfx/gen-1-1-5.expected.jsonl` record for record,
  and the last passage is `vulgata-clementina:REV.22.21` with text `Gratia Domini nostri
  Jesu Christi cum omnibus vobis. Amen.`
- The output file's `sha256` equals the committed digest in
  `tests/fixtures/corpus/usfx/vulgate.digest.json` (`{"sha256", "bytes", "passages"}`);
  `corpus validate` logic (`corpus_validation`) over the same output reports no findings.
- The fixture records its wall-clock duration via `record_property("normalize_seconds",
  ...)` against the 15 s CI budget, asserts it is under 60 s, and the module is skipped when
  `MUTANT_UNDER_TEST` is set (`pytest.mark.skipif`).
- Assert (do not add) that `[tool.mutmut].also_copy` lists `data/` (added by T-102, whose
  `tests/unit/mutation_config_test.py` and `scripts/check-mutation-config-test.sh` already
  assert it).
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the measured normalization duration and the committed output digest.

## Implementation Notes

- The digest file is produced once by `scripts/regen-corpus-fixtures.py --target
  vulgate-digest` (T-104) and committed; the test never writes it. A digest change
  requires a deliberate regen and a commit explaining why.
- Keep the regex oracle deliberately simple and separate from the adapter: it must not
  share code, constants or helpers with `usfx_adapter.py`.
- Do not add a second full-corpus test (legacy full-Vulgate digest from T-102 is the only
  other reader of the file, through the legacy path).
- If the oracle disagrees with the adapter, fix the adapter with a minimal synthetic
  regression fixture in `tests/fixtures/corpus/usfx/` and keep the oracle unchanged.
- Size guard: one test module, one digest file and the mutation-config assertions. File a
  follow-up rather than widen.

## Execution Evidence

- Accepted base: `origin/main` at `7a22673d4864f751bc2a1a51e613fa1b7e27dc49`;
  v0.3.0 spec SHA-256 remained
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
  Validation and selector confirmed this task with no conflicting owner before start
  at 2026-10-16T07:58:14Z.
- Explicitly authorized size-guard exception: add the missing `vulgate-digest`
  target to the existing fixture generator. The prescribed capture command rejected
  that choice on the accepted base, so this small extension is necessary to follow
  the Implementation Notes. Existing symlink/overwrite preflight and atomic output
  conventions are preserved. No acceptance, spec, priority, dependency or gate changed.
- Strict TDD: the new module initially failed twice: the five-verse generator test
  raised argparse `SystemExit: 2` for the missing target; full acceptance passed the
  regex/count/literal checks but failed on the absent digest. After implementation,
  the generator regression passed; the prescribed regeneration command captured the
  digest; both module tests passed. Generator regression uses only the small Genesis
  fixture, not a second full-corpus normalization.
- Output: 24,905,456 bytes, 35,809 passages, SHA-256
  `cefcaa8f977df72acd71b76ace7a45ccbe8e81105dd56a86815964e531e0927d`.
  Initial recorded normalization: 4.621132505 seconds (15-second CI budget,
  60-second asserted ceiling). Source hash before/after matches the pinned value;
  73 works, 1,334 chapters, independent IDs/per-book counts, GEN/REV literals,
  first-five passage golden and full output validation pass. Adapter unchanged.
- Initial exact ruff/mypy/pytest chain passed: 1,431 tests, one existing Pydantic
  warning from the float-rejection test. Mutation environment probe skipped both
  module tests; existing `data/` copy configuration was asserted, not changed.
- Dedicated code-simplifier Task loaded the personal skill: no useful simplification;
  scoped ruff and both tests passed. Separate parallel read-only code-reviewer Tasks
  loaded General (ECC code-reviewer/common guidance), Python (ECC python-reviewer
  and python-patterns), and Security (ECC security-reviewer/security-review guidance).
  Security covers fixture-write safety; database/framework lanes do not apply.
- Every lane reported verbatim: "No concrete task-relevant findings." Fresh
  candidate-validation Task loaded code-reviewer, inspected the actual diff and
  confirmed the conclusion; no candidate IDs, rejections, deduplication or deferred
  findings. Focused tests, ruff and mypy passed in validation. No finding fixes needed.
  One review/disposition cycle; fresh code-reviewer disposition verification confirmed
  no unresolved or new task-relevant findings after inspecting final gate logs.
- Final exact `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain passed:
  140 source files type-checked, 1,431 tests passed in 91.42 seconds. `mise run check`
  passed formatting, ruff, mypy, all 1,431 tests in 84.65 seconds, and commit-message,
  push-message, author, mutation-results and mutation-configuration guards. Both full
  runs reported only the existing float-rejection Pydantic warning. Taskrail verify
  passed at 2026-10-16T08:11:46Z; complete at 2026-10-16T08:11:54Z. State validates;
  observed next selector is T-112 (not started in this cycle).
- Scope exclusions preserved: T-112 owns legacy integration; T-127 owns schema
  exporter repair; future v0.8.0 T-128 remains unstarted with its prior disposition,
  not implemented or reopened here. No follow-up needed for this task.
- 2026-10-16T08:11:46Z: verification pass
- 2026-10-16T08:11:54Z: Verified 2026-10-16T08:11:46Z after exact ruff/mypy/pytest and mise run check passed; independent General/Python/Security, fresh candidate and disposition checks found no concrete issues. Source and spec unchanged; authorized prescribed digest generator exception documented.
