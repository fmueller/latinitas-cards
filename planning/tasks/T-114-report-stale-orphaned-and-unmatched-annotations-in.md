---
id: T-114-report-stale-orphaned-and-unmatched-annotations-in
title: Report stale, orphaned and unmatched annotations in inspect and validate
status: completed
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies:
    - T-113-define-annotation-sidecar-records-and-the
    - T-107-normalize-plain-text-sources-with-manifest
updated_at: "2026-10-16T09:51:25Z"
---

# T-114-report-stale-orphaned-and-unmatched-annotations-in Report stale, orphaned and unmatched annotations in inspect and validate

## Description

Report anchor states from `specs/v0.3.0.md#sidecar-annotation-model` and Fixed Contracts
**Identity and hashing** ("hash mismatch is `stale`, superseded segmentation `orphaned`,
missing passage `unmatched`") through `annotations inspect SIDECAR --corpus CORPUS
[--check] [--json]` and `corpus validate CORPUS --annotations SIDECAR`, and omit such
annotations from `corpus show CORPUS [--passage ID]... [--limit N] [--annotations
SIDECAR]`. Nothing is ever re-anchored. It builds on the sidecar model (T-113) and uses
plain-text fixtures normalized with `--citation-template` (T-107). No other task adds `corpus
show`; this task adds it to `commands/corpus.py` (from T-105). Not in this task: `--claims`
on any command (T-125), decision staleness (T-115), dependency-aware plan routing (T-117).

## Acceptance

- `src/latinitas_cards/annotation_state.py` exposes `annotation_states(corpus, sidecar)`
  returning, per `annotation_id`, an anchor state `current|stale|orphaned|unmatched` and
  lifecycle flags `superseded`/`retracted`, plus `dependents(sidecar, annotation_id)`
  (records whose anchor `segmentation_id` names it). `unmatched`: `passage_id` absent from
  the corpus; `stale`: anchor `passage_hash` differs from the corpus passage; `orphaned`:
  `segmentation_id` names a segmentation record that a later record supersedes. Unit tests
  cover each state and an unchanged control.
- Fixtures: `tests/fixtures/corpus/text/wsl-three.txt` (three `LABEL<TAB>text` lines),
  `text/wsl-three-edited.txt` (one word changed in line 2) and `text/wsl-two.txt` (line 3
  removed), normalized with `corpus normalize ... --adapter text --citation-template
  wsl:work.section.line`, and `tests/fixtures/annotations/` JSONL fragments of morphology
  records anchored to line tokens; commit their normalized corpora as
  `tests/fixtures/corpus/wsl-three.corpus.jsonl` and `wsl-three-edited.corpus.jsonl` for
  T-115. Flow: normalize, `annotations segment --timestamp 2026-01-01T00:00:00Z`, append the
  morphology records with `annotation_sidecar.append_records`,
  re-normalize the edited source to the same corpus path; `annotations inspect SIDECAR
  --corpus CORPUS --check --json` exits 1 with exactly the line-2 annotations reported
  `stale` (`ref` = `annotation_id`), lines 1 and 3 `current`; without `--check` it exits 0
  with the same findings as warnings; `corpus validate CORPUS --annotations SIDECAR
  --json` exits 1 with the same codes. The unedited control exits 0 with no diagnostics in
  both commands.
- Re-normalizing `wsl-two.txt` reports line 3 annotations `unmatched` (exit 1 with
  `--check`). A committed sidecar where a later record supersedes line 1's segmentation
  reports line 1's token-anchored morphology `orphaned` while line 1's passage-level
  records and all line 2 and 3 records stay `current`.
- `annotations inspect --json` lists for every record its `annotation_id`, layer, state,
  flags, `passage_id`, the corpus header `source.sha256`, producer, `plan` and `created`,
  plus `input_hash`, in sidecar order; text output escapes controls. This is the
  provenance surface consumed by T-119; it requires no claim join.
- `corpus show CORPUS --annotations SIDECAR --json` returns passages in corpus order
  (default `--limit 20`, `--limit 1000` accepted; `--limit` is a Click `IntRange(1, 1000)`,
  so `--limit 1001` and `--limit 0` exit 2; repeated `--passage ID` selects those IDs and an
  unknown ID exits 3 `unknown_passage`) with only `current`, non-superseded, non-retracted
  annotations: after the edit flow the line-2 annotations are absent and lines 1 and 3
  unchanged. `result = {"passages": [{"passage_id", "citation", "text", "annotations":
  [{"annotation_id", "layer", "value", "producer"}]}], "truncated": bool}`.
- Read-only: `annotations inspect`, `corpus validate` and `corpus show` leave corpus and
  sidecar bytes unchanged and create no `.lock` file; an unsupported or invalid sidecar
  exits 3 (`unsupported_format`, `unsupported_version`, `invalid_sidecar`). Matrix rows:
  `annotations inspect: 0, 1, 3`, `corpus show: 0, 3`, and `corpus validate` gains its
  `--annotations` triggers.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the `--check` JSON of the edit, removal and supersede fixtures.

### Workflow-v3 evidence (2026-10-16)

- Base HEAD and fetched origin/main both matched
  `1f7c1223620b3b2364db4c1c3724e0f8f3a132ca`; pinned spec SHA256 matched
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
  `mise exec -- taskrail validate` reported `state valid`; `next --json` selected
  T-114 with `off_spec: false` before start at 2026-10-16T09:29:26Z.
- Read the complete task. Relevant bounded spec inspection covered sidecars
  (148-190), identity/version contracts (349-407), CLI result (192-213), and
  read-only commands/exit rules (480-504), not the complete spec. The initial
  denied whole-spec IO Delegate request was not bypassed; subsequent bounded
  native reads followed the clarified authorization. No provider settings changed.
- Strict TDD order: `uv run pytest -q tests/unit/annotation_state_test.py`
  failed with missing module, then passed; inspect test failed `No such command
  'inspect'`, then passed; validation tests failed `No such option: --annotations`,
  then both passed; show test failed `No such command 'show'`, then passed.
  A further state test proved earlier forward supersession must not flag a later
  proposal: red `assert True is False`, then green after honoring stream order.
- Real normalization/segment/append/edit, removal and supersession workflows
  passed, including byte-for-byte comparisons to the committed corpora and
  sidecars. Readers left bytes/mtimes unchanged and created no locks. The actual
  subprocess probes captured full `--check --json` envelopes at
  2026-10-16T09:46:04Z; control exit 0/no diagnostics, edit exit 1/two stale,
  removal exit 1/two unmatched, supersession exit 1/one orphaned. Summaries:

  ```json
  {"edit":{"exit_code":1,"refs":["sha256:5db20f7ec29d3d7a5ff1bcdb8e1c762e6e80440f293989ccfdc24ffa2e9b1514","sha256:4d5110efa95bf9d0e84e7ac32fbd71086b43e3d01f3243d20a49cb96e9f04041"],"code":"stale"},"removal":{"exit_code":1,"refs":["sha256:7476dda82d28b33546a9734a63440fbaf827f95ce337fff421cbd736a81552df","sha256:440bfda602c2ab0bc9ef8301577a8a7933a7f08006aa35c1d3dde351e98404c4"],"code":"unmatched"},"supersede":{"exit_code":1,"refs":["sha256:5d69255e76a21cea0e47446897fc64ede7a2e661de28c79d454401396494609a"],"code":"orphaned"}}
  ```

- Show annotation counts by passage after editing: `[2,0,2]`; removal: `[2,2]`;
  supersession: `[1,2,2]`. Unaffected records remain current. Warnings without
  `--check` and validation error diagnostics are compared verbatim in tests.
  Coverage includes provenance, retractions, control escaping, duplicate/repeated
  passage IDs, unknown IDs, default 20/maximum 1000, invalid limits, truncated
  selection, invalid corpus tail, and unsupported/invalid sidecars.
- Dedicated Task loaded `code-simplifier`: precomputed inactive segmentation IDs
  without changing behavior; rejected broader restructuring. Parent post-pass
  focused suite: 140 passed. Fixture regeneration SHA256 recheck: all five outputs
  unchanged. Scratch inspection probe: one corpus read, three passages, peak two
  live passage objects, zero retained; no dense-memory overhead claim is made.
- Separate parallel read-only Tasks loaded `code-reviewer` for General (ECC code
  reviewer, no companions), Python (ECC Python reviewer, Python patterns/common
  rules), Security (ECC security reviewer, security-review/cloud guidance).
  General/Python each ran 171 focused tests; Python also reran full gates;
  Security ran 120 tests. Database/framework/domain lanes omitted because no DB,
  framework or ML/network implementation changes. Three lanes fit the budget.
- Each review returned verbatim: **"No concrete task-relevant findings."** Fresh
  candidate-validation Task loaded `code-reviewer` and confirmed the same result:
  zero candidates, zero validated/rejected findings, zero dispositions/deferrals.
  Fresh disposition-verification Task loaded `code-reviewer` after final gates and
  returned verbatim: **"No concrete task-relevant findings."** One review cycle;
  no review-driven fixes or fabricated finding rows.
- Final exact chain passed: `uv run ruff check` (all checks passed), `uv run mypy`
  (148 source files), `uv run pytest -v` (1563 passed, one existing Pydantic
  serializer warning, 102.43s). `mise run check` exited 0, including formatting,
  full pytest, commit/push message, author and mutation guard suites. Initial
  full-suite failure was only the old exact command-registration expectation;
  it now includes inspect/show and the full chain was rerun from the beginning.
- Scope: one new state module, callbacks in existing command modules, necessary
  unknown-passage registry/schema addition, fixture generator target, tests/docs.
  T-127 remains todo and its exporter is not repaired. Claims, decisions,
  planning, policy, enriched export, T-038 and future T-128 are untouched.

## Implementation Notes

- New `annotation_state.py` for the state logic; `annotations inspect` goes in
  `commands/annotations.py`; `corpus validate --annotations` and `corpus show` extend
  `commands/corpus.py` (from T-105) through a thin callback that calls `annotation_state`.
  Use `run_contract` and registry codes `stale`, `orphaned`, `unmatched`.
- A record has one anchor state, checked in the order `unmatched`, `stale`, `orphaned`. An
  unknown `segmentation_id` and a retracted segmentation are not fixed by the spec: treat
  both as `orphaned`. This is the v0.3.0 rule; document it in `docs/annotation-sidecars.md`.
- Streaming: read the corpus once into a `passage_id -> passage_hash` map; never hold
  passage text for `inspect`.
- `corpus show` is a bounded read-only view (no export format, no `--output`), per
  `#sidecar-annotation-model`; v0.6.0 owns enriched export.
- TDD order: state unit tests, inspect, validate, show. Size guard: one module plus
  callbacks in two existing command modules.
- 2026-10-16T09:51:17Z: verification pass
- 2026-10-16T09:51:25Z: Reviewed workflow-v3 implementation complete. Verification passed at 2026-10-16T09:51:17Z after ruff/mypy/1563 pytest and mise run check; General/Python/Security and fresh candidate/disposition verifiers found no concrete task-relevant findings.
