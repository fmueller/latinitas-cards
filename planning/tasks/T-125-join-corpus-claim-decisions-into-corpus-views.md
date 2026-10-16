---
id: T-125-join-corpus-claim-decisions-into-corpus-views
title: Join corpus claim decisions into corpus show, corpus validate and annotations inspect
status: in_progress
priority: medium
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies:
    - T-114-report-stale-orphaned-and-unmatched-annotations-in
    - T-115-record-corpus-claim-decisions-through-claim
updated_at: "2026-10-16T14:16:38Z"
---

# T-125-join-corpus-claim-decisions-into-corpus-views Join corpus claim decisions into corpus show, corpus validate and annotations inspect

## Description

Join corpus claim decisions into the read-only corpus views of
`specs/v0.3.0.md#sidecar-annotation-model` ("A corpus annotation accepted through `claims
record` shows as accepted in `claims inspect` and `corpus show` and turns stale with it")
through the `[--claims SIDECAR]` options that Fixed Contracts **Commands** gives `corpus
validate`, `corpus show` and `annotations inspect`. It builds on the anchor states,
`annotations inspect`, `corpus show` and `corpus validate --annotations` of T-114 and on
the version 2 claim sidecar and `corpus_claims` of T-115. It adds the `--claims SIDECAR`
option to those three commands and a join helper; no other command or option is added. Not
in this task: decision-aware planning/apply safety (T-117, T-119) and any change to `claims
list|record|inspect` (T-115).

## Acceptance

- Review joins (staleness fixtures from T-114, `tests/fixtures/corpus/wsl-three.corpus.jsonl`
  and `wsl-three-edited.corpus.jsonl`): after `claims record` accepts a line-2 morphology
  annotation, `claims inspect` and `corpus show CORPUS --annotations SIDECAR --claims
  CORPUS.claims.json --json` show it accepted and `annotations inspect SIDECAR --corpus
  CORPUS --claims CORPUS.claims.json --json` reports `accepted` for that record.
- After the one-word edit and re-normalization the claim turns stale with its annotation:
  `claims inspect` reports the decision `stale`, `corpus show` omits it, `corpus validate
  CORPUS --annotations SIDECAR --claims CORPUS.claims.json --json` exits 1 and reports the
  stale decision with code `stale`, `ref` = the claim fingerprint, and the line 1 and 3
  decisions stay accepted. The unedited control exits 0 with no diagnostics.
- In `corpus show` and `annotations inspect`, an annotation with no decision is reported
  with `review_status` `open` (the `claims list` value set `open|accepted|withheld`); no
  decision is created or changed by any of the three commands.
- On `corpus show`/`corpus validate`, `--claims` without `--annotations` exits 2
  `usage_error` (decisions bind annotations). `annotations inspect` already takes its
  annotation SIDECAR positionally; it must not require a nonexistent `--annotations` flag.
  A `--claims` file that is missing or unreadable exits 3 `unreadable`; a version 1 or
  higher-version document exits 3 `unsupported_version`, both with one `CliResult` envelope
  under `--json`. The three commands stay read-only: corpus, sidecar and claims file bytes
  are unchanged and no `.lock` file is created.
- The `corpus show`, `corpus validate` and `annotations inspect` rows of
  `docs/agent-cli-contract.md` gain their `--claims` triggers, each with a
  `CONTRACT_EXIT_CASES` entry, and the registry-equals-docs test passes.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the `corpus show --claims` and `corpus validate --claims` JSON before and after
  the edit, and the SHA-256 of the claims file before and after each read-only run.

## Implementation Notes

- Add one join helper to `corpus_claims.py` (decision lookup by corpus claim fingerprint,
  reusing `current_annotations` and the T-115 v2 loader); the three commands call it
  through thin `--claims` options in `commands/corpus.py` and `commands/annotations.py`,
  using `run_contract` (T-103).
- Load the claims file through `claim_sidecar` read-only, without taking its lock.
- Update `docs/reviewed-claim-cli.md` with the `--claims` joins (corpus mode section from
  T-115).
- TDD order: join helper, `corpus show --claims`, `annotations inspect --claims`, `corpus
  validate --claims`. Size guard: `--claims` options on three existing callbacks plus a join
  helper in `corpus_claims.py`; a missing capability elsewhere becomes a follow-up task.

## Workflow-v3 audit (2026-10-16)

### Understand and ownership

- Fetched origin/main before lifecycle writes; HEAD and remote both matched accepted
  cumulative base `c90e20937d4668d56ac6cf1457c2f3d05eb34654`. Worktree initially clean.
- Active spec `specs/v0.3.0.md` SHA256:
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
- `mise exec -- taskrail validate`: `state valid`; `next --json` selected this task,
  todo, with zero active. Started only T-125 using Taskrail.
- Ownership: three existing callbacks and one join helper; keep T114 states,
  T115 claim binding/fingerprint, v1 deck workflows and T117–119 safety unchanged.
  No planning/apply, exporter repair, spec, priority, dependency or gate changes.

### Strict TDD and initial checks

| Order | Red evidence | Green evidence |
| --- | --- | --- |
| Helper | `uv run pytest tests/unit/corpus_claims_test.py -k test_join_corpus_decisions -q`: ImportError for absent helper | 1 passed after helper; an intermediate failure caught incorrect Diagnostic field access (`index`, corrected to existing `item`) |
| Show | `-k test_claim_view_join`: no such option `--claims`, exit 2 | 1 passed |
| Inspect | `-k 'test_claim_view_join and inspect'`: no such option, exit 2 | 1 passed |
| Validate | `-k 'test_claim_view_join and validate'`: no such option, exit 2 | all four helper/view tests passed |

- Added observable coverage for open, withheld, accepted, stale and unaffected
  passage controls; usage2, missing/directory unreadable3, v1/v3 version3,
  one-envelope parsing and unchanged bytes/mtimes/no locks.
- Initial exact chain `uv run ruff check` → `uv run mypy` → `uv run pytest -v`
  first caught import sorting, then type annotations. Fixed and restarted the
  full chain each time: all checks passed; 169 files typechecked; 2475 passed,
  one existing Pydantic serializer warning.

### Simplification and independent review

- Dedicated Task loaded personal `code-simplifier`; replaced walrus comprehension
  with explicit status loop and type annotation. Accepted, no other edits.
  Focused checks and 93 tests passed; parent inspected the diff.
- Separate parallel read-only Tasks loaded personal `code-reviewer` and index:
  General: ECC code-reviewer, no companions; Python: ECC python-reviewer plus
  python-patterns; Security: ECC security-reviewer plus security-review and
  common security/code-review rules. Security independently ran 43 claim tests.
- General required for correctness; Python for changed Python code; Security for
  untrusted explicit file inputs and fail-closed review binding. Database omitted:
  no SQL/migration/database client, persistence or locking change; existing lock
  safety reviewed in General/Security. Framework and other domains not involved.
- Python and Security each returned: "No concrete task-relevant findings."
- Fresh candidate-validation Task loaded `code-reviewer`, checked actual diff and
  executed both invalid-corpus probes; validated G-1, no duplicates/rejections.

### Verbatim finding and disposition

> G-1: When a corpus has validation findings, using `--claims` in `corpus validate`
> or `annotations inspect` can turn the expected validation diagnostics into an
> unexpected internal-error envelope (exit 70).

- Evidence: helper called `current_annotations` outside loader catch; that raises
  on corpus validation diagnostics, despite callers already collecting them.
- Fixed in scope: skip joins on corpus errors, preserving original diagnostics;
  annotation stale errors still join so stale claim refs remain reported.
- Finding-fix TDD: `uv run pytest tests/unit/corpus_claims_test.py -k preserves_invalid -q`
  failed twice with exit70 vs expected1 before fix. After fix focused claim/matrix
  suite passed 95 tests, including both regression cases. No findings deferred.
- Fresh exact chain after fix: ruff passed, mypy passed (169 files), full pytest
  2477 passed, one existing Pydantic serializer warning. Full suites sequential.

### Manual read-only evidence

- Durable literal outputs and argv are in `planning/T-125-read-only-evidence.json`.
  Eight real CLI reads cover claims inspect, show, inspect, validate before/after
  normalizing the one-word edited text with `wsl:work.section.line` citations.
  Temporary normalization and recording locks were removed before reads;
  every read asserts no locks are created and unchanged corpus/annotation/claims
  bytes and mtimes. Per-input byte lengths, SHA256 and mtime pairs are retained.
- All six decisions explicitly accepted via `claims record` before edit. Line2
  morphology accepted before, stale afterward; show omits it, validate exits1
  `stale` with fingerprint ref. Line1/3 morphology controls stay accepted; unedited
  validation exits0 without diagnostics. Unit tests separately cover open/withheld.
- Claims SHA256 before and after every read:
  `a5618c0791135c8209ce755d592195f1efa16dedb6012854c7d3a178169299a1`.
- Docs cover joins, triggers, exit matrix and thin run_contract callbacks; registry
  equality test passes. No decision writes, automatic acceptance or reader locks.

### Final gates and disposition verification

- `mise run check` exited0: ruff, format, mypy, 2477 pytest tests and all commit,
  push, author, mutation, distribution (54 cases) and publish-guard checks passed.
- Fresh read-only disposition-verification Task loaded `code-reviewer` and
  General/Python/Security references, inspected updated diff and evidence,
  independently reran the two G-1 regressions (2 passed, 43 deselected).
  Its final conclusion: "G-1 is resolved; no other concrete task-relevant findings
  or deferred findings remain." One review/fix/recheck cycle; none unresolved.
- Spec pin rechecked unchanged, Taskrail `state valid`, only T125 active before
  finalization. Hooks installed before first commit; maintainer identity is
  Felix Müller. No attribution trailers or external release actions.
