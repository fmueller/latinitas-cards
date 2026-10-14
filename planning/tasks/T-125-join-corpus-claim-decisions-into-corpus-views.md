---
id: T-125-join-corpus-claim-decisions-into-corpus-views
title: Join corpus claim decisions into corpus show, corpus validate and annotations inspect
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies:
    - T-114-report-stale-orphaned-and-unmatched-annotations-in
    - T-115-record-corpus-claim-decisions-through-claim
updated_at: "2026-10-14T11:31:03Z"
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
