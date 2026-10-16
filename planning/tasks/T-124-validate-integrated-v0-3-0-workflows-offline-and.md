---
id: T-124-validate-integrated-v0-3-0-workflows-offline-and
title: Validate integrated v0.3.0 workflows offline and prepare the 0.3.0 candidate
status: completed
priority: low
spec_ref: specs/v0.3.0.md#goals
dependencies:
    - T-112-route-legacy-vulgate-commands-through-the-adapter
    - T-114-report-stale-orphaned-and-unmatched-annotations-in
    - T-115-record-corpus-claim-decisions-through-claim
    - T-119-enforce-destructive-conflicting-and-actor-rules
    - T-120-report-effective-configuration-and-guarantee-no
    - T-123-document-pypi-installation-extras-and-the
    - T-125-join-corpus-claim-decisions-into-corpus-views
    - T-126-prepare-v0-3-0-orchestrated-loop
updated_at: "2026-10-16T15:25:03Z"
---

# T-124-validate-integrated-v0-3-0-workflows-offline-and Validate integrated v0.3.0 workflows offline and prepare the 0.3.0 candidate

## Description

Combines the T-089 (integrated validation) and T-100 (candidate metadata) roles for
v0.3.0: Delivery Order step 7 of `specs/v0.3.0.md` against `#goals`. It runs an offline
end-to-end walkthrough over every v0.3.0 feature using only documented CLI commands,
then prepares the 0.3.0 candidate (version, lock, changelog, readiness, README). Because
v0.3.0 changes no Anki-facing output (Summary), the readiness record cites no Anki,
native Desktop or AnkiMobile evidence and reopens no v0.2.x native gate. Tagging,
GitHub release and PyPI publication belong to the maintainer gate (T-038), never to this
task; runtime code is not changed here. Activating v0.3.0 precedes the loop (operator);
activating v0.4.0 is out of scope for both this task and T-038.

## Acceptance

- At start and immediately before `taskrail verify`, `taskrail status --json` shows
  `active_spec_path` `specs/v0.3.0.md`, `active_spec_drift.active_open_count` equal to 2
  (this task active, T-038 blocked) and `blocked` listing only T-038. Tasks on other specs
  (`active_spec_drift.away`) are ignored. Otherwise: if every other open v0.3.0 task is
  `blocked`, `taskrail block` this task naming each one (a maintainer decision is needed).
  If any open v0.3.0 task is `todo`, requeue this task as described below.
- T-126 preparation must be completed before this task starts; it is not an additional
  exception to the all-other-work guard. Preserve T-038 as the only blocked maintainer
  release gate. An unexpected other `in_progress` task is not readiness: stop and report
  the lifecycle inconsistency without overriding the single-active-task constraint.
- In a temp dir outside the checkout, a walkthrough of documented commands exits as
  documented: `corpus normalize <repo>/data/lat-clementine.usfx.xml --adapter usfx --corpus-id
  vulgata-clementina --descriptor data/lat-clementine.descriptor.json --output
  vulgata.jsonl`; the committed
  plain-text fixture through a manifest (first run, then an edited copy resolved with
  `--approve-reuse`/`--approve-allocation`/`--approve-removal`) and through
  `--citation-template`; `corpus validate --json` (0); `annotations segment --timestamp`;
  a `fixture-morph/1` patch through `annotations plan`, `approve --actor human:<id>` and
  `apply`, then a rerun reporting `already_applied`; `claims record` accepting one
  corpus annotation, shown by `claims inspect` and `corpus show`; `annotations inspect
  --check` (0); `corpus config --json`; `policy validate`.
- Negative controls in the same walkthrough: one edited passage word makes its
  annotations `stale` (`annotations inspect --check` and `corpus validate --annotations`
  exit 1) and the accepted claim stale; a foreign sidecar change makes `apply` exit 4
  with no byte change; an `agent:` receipt under the default policy exits 5
  (`actor_refused`); a DTD input exits 3 (`unsafe_xml`); `--approve-reuse` of a retired
  ref exits 3. A second fresh directory with different `TZ` and locale reproduces every
  normalize, segment, plan and apply output SHA-256, and the Vulgate output digest equals
  the committed full-corpus oracle digest.
- Prepared-contract controls: normalize templated refs containing raw `%2E` and `é`,
  validate the encoded `%252E`/`%C3%A9` spellings and raw citation values; segment both
  `d'arma` and `d’arma` into word/punctuation/word tokens. Exercise legacy claims JSON
  and exit exceptions (forbidden corpus options: 3; invalid batch: 1, no writes).
  Plan/approve a retract, then record an accepted and separately withheld decision;
  old-receipt apply exits 4 `binding_drift`, unchanged sidecar. An unrelated decision
  change permits apply, and unchanged partial/re-receipted/no-op retries remain valid.
  Cite T-107's interruption/recovery evidence (before/between/after replacements,
  first creation, original approved rerun, no double allocation), and T-118/T-119's
  lock/concurrent-claim and same-plan-dependent/conflict regression results. Do not
  duplicate their fault-injection machinery or claim two-path atomic visibility.
- Golden files committed by T-102 are unchanged since the commit that added them: `git
  diff $(git log --diff-filter=A --format=%H -1 --
  tests/fixtures/goldens/legacy-vulgate/cases.json) -- tests/fixtures/goldens/legacy-vulgate
  tests/fixtures/goldens/claims-v1` is empty, and their tests pass.
- On the bumped candidate run `UV_OFFLINE=1 bash scripts/check-dist.sh` and `UV_OFFLINE=1
  bash scripts/check-install.sh "$(uv python find 3.14)"` against the cache warmed by `uv
  sync --locked --dev`; both report version 0.3.0. If either fails only because a package
  is not cached, rerun without `UV_OFFLINE`. If no network is available, record
  "check-install not run: offline" with the error line and continue. The `package` CI job
  (verified on the candidate by T-038) is the binding check, and the offline
  `check-dist-test.sh` in `mise run check` remains mandatory.
- TDD: rename and update `test_release_metadata_and_current_lock_audit_are_versioned_for_v021`
  in `tests/unit/release_candidate_test.py` first (RED on `'0.2.1' == '0.3.0'`), then
  set `project.version` and the `uv.lock` `latinitas-cards` entry to 0.3.0 with an
  unchanged dependency inventory (the 92-record audit assertions pass unchanged), move
  pending entries under `## [0.3.0] - YYYY-MM-DD` with an empty `[Unreleased]`, and add
  `compare/v0.3.0...HEAD` and `compare/v0.2.1...v0.3.0` links. Historical v0.1.x and
  v0.2.x assertions stay.
- New `docs/release-v0.3.0.md` records scope, walkthrough commands, exit codes and
  digests, and limits (experimental `annotate` extras and the pip CPU-index caveat,
  format version 1 before 1.0, Linux and macOS only, Vulgate provenance pending the
  maintainer's confirmation, no native Anki or Mobile evidence because Anki-facing output
  is unchanged); `docs/release-readiness.md` and the README Documentation list link it;
  README "**Current release:**" and the git checkout command name v0.3.0. Tests assert
  these strings and that README stays free of relative links.
- No tag, GitHub release, PyPI upload, workflow dispatch, spec activation or runtime
  change; a product defect found by the walkthrough is filed with `taskrail task new
  --spec-ref specs/v0.3.0.md#<anchor of the defective feature> --priority high --title
  "..."` (never `--follow-up` on this task, which would make the fix depend on this task).
  This task is then requeued with `taskrail block <this id> --reason "requeued behind
  T-NNN"` followed by `taskrail unblock <this id> --reason "requeued behind T-NNN"`, and
  the cycle reports "requeued". Because this task is `low` and the fix is `high`, the
  selector runs the fix first. Uncommitted candidate edits (version, changelog) are
  discarded before requeueing.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` and `taskrail validate` pass.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the candidate base commit, the pinned `specs/v0.3.0.md` SHA-256, every
  walkthrough command with its exit code, output SHA-256 values, and the local
  `check-dist.sh`/`check-install.sh` artifact names and hashes.

## Implementation Notes

- Follow T-089 and T-100: documentation and validation, no new production code, no
  artificial TDD for prose. The walkthrough is a recorded CLI session, not a second
  full-corpus pytest test (the T-111 test is the only one).
- Use committed fixtures from T-107, T-109, T-115 and T-118; do not build Python objects.
  Run commands as `uv run --project <checkout> latinitas-cards ...` (or
  `<checkout>/.venv/bin/latinitas-cards`) from the temp dir (cwd stays the temp dir) with
  absolute input paths.
- Changelog: write short user-focused `Added`/`Changed`/`Limitations` bullets for every
  v0.3.0 capability from the task titles and docs (feature tasks add no changelog entries;
  only T-123 pre-adds one, its PyPI bullet, which is folded in); per AGENTS.md no task IDs
  or evidence. The date is the candidate date; T-038 may correct only that date before
  tagging.
- v0.3.0 follow-up rule for implementation tasks (not this validation-only task). (a) A
  defect in completed work that the current implementation task builds on is fixed
  inside that task, with a regression test, not filed as a follow-up: the
  selector would run the follow-up only after every remaining higher-priority or lower-id
  task. (b) Follow-ups are filed only for work that no remaining v0.3.0 task needs. They
  use `--priority medium` or `high`, never `low`, so they run before T-124. (c) A follow-up
  that needs a maintainer decision (for example a Fixed Contracts disagreement) is created
  and then `taskrail block`ed with the reason "maintainer decision: ...". T-124 will then
  block at its start check. That is the intended stop.
- The autonomous loop runs `/workflow-loop-v0` with the gate `through T-124 inclusive`
  (not `all-open`), so the permanently blocked T-038 is never reported as a failure and a
  requeued T-124 keeps the gate open; a "requeued" T-124 cycle is an accepted outcome.
- `docs/release-readiness.md` is the v0.1.0 record whose strings are asserted; add a
  short pointer to the current record without rewriting the historical content.
- Leave `specs/README.md` and Taskrail spec activation untouched.
- Size guard: metadata, two docs and one existing test. If the walkthrough exposes
  product gaps, file them as described in Acceptance (never `--follow-up` on this task)
  rather than fix them here.

## Workflow-v3 execution evidence (2026-10-16)

- Understand/start: fetched origin/main; HEAD and remote both
  `1db086cae196dafa4a36350bdd354b46e9b66ccc`. Initial T127 ancestry lookup
  lacked the shallow object; unshallowed before golden history inspection and
  successfully verified containment of T127 implementation
  `e2fa794e7dce41e3a09e35ed3b7190b2b366741a`. Clean initial worktree;
  no source-overseer local edits transferred. Spec SHA-256 stayed
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
  `mise exec -- taskrail validate` passed and `next --json` selected this task
  as todo. Start guard: active-spec open count 2, blocked only T038, active 0;
  T126/T127 completed and T128 away on v0.8.0. Started only T124, then status
  confirmed active 1/open 2/only T038 blocked. No other lifecycle owner existed.
- Walkthrough: [candidate record](../../docs/release-v0.3.0.md) contains every
  exact CLI argv/exit, observable summaries and output digests. The subprocess
  capture SHA-256 is
  `a835ba98a04a83c7b48f03a54472fd37b61d23183c49fc03e6e5d797ed24e00a`.
  Full Vulgate oracle matched 35,809 passages/24,905,456 bytes. Plain manifest
  first/edit/review approvals, labeled template, percent/Unicode refs, both
  apostrophes, segment/fixture-morph plan/human approve/apply/rerun, explicit
  accepted claim and joined show/inspect, config/policy all passed. Negative
  stale/foreign suffix/default-agent/DTD/retired-ref and legacy claims option/
  batch controls returned prescribed 1/4/5/3 codes and preserved protected bytes.
  Independent accepted and withheld post-plan decisions rejected old receipts;
  unrelated decisions, partial retries, later/different receipts and completed
  no-ops remained valid. TZ UTC/C.UTF-8 versus Honolulu/C in a second fresh
  directory reproduced normalize/segment/plan/receipt/apply hashes. Harness-only
  mistakes (unsupported config argv, conflicting decision overwrite) were
  corrected using documented commands and separate copies, retained in transcript.
  No product defect found; no runtime edits or high-priority follow-up/requeue.
- Prior evidence: cited T107 before/between/after replacement and first-creation
  approved-request recovery/no double allocation, and T118/T119 exact claims-lock,
  concurrent claims writer, same-plan dependent/conflict and drift regressions.
  No duplicate fault machinery, two-path atomic visibility or native acceptance
  claim. Historical golden diff from T102 addition was empty after unshallowing.
- TDD: renamed release metadata test before version edits. Targeted pytest RED:
  `AssertionError: assert '0.2.1' == '0.3.0'`; after metadata/docs edits GREEN:
  1 passed, 2 deselected. First full chain exposed two candidate-test assumptions:
  install docs required the old clone command/nonempty Unreleased, and package
  inventory included the project version. Minimal expectation updates in those
  two existing tests were necessary beyond the one-test size estimate, not runtime
  work. Subsequent focused candidate/install/package tests: 9 passed. The 92-record
  release audit stayed unchanged; non-project lock SHA-256 before/after equals
  `4023b72c678c2b3312f407a7bc69d3434cdc803e6682e9e023aafb9d28bf4a89`.
- Initial checks: exact `uv run ruff check` -> `uv run mypy` -> `uv run pytest -v`
  restarted after the two test failures; passed 169 source files and 2,478 tests
  (113.59s), one existing Pydantic serialization warning. `git diff --check` passed.
  Packaging on bumped candidate: offline check-dist 0; offline check-install 1
  solely uncached pydantic; authorized network retry 0 for wheel and sdist,
  CPython 3.14.2, no optional NLP distributions. Artifact names/hashes and exact
  commands are in the candidate record. No publication claim; package CI is binding.
- Simplify: dedicated Task loaded personal code-simplifier, inspected the diff,
  made no edits, and reported 9 focused tests passed plus diff check. No rejected
  simplifications. No production code changed.
- Independent review: separate parallel read-only Tasks loaded code-reviewer and
  reviewer-index mappings. General loaded ECC code-reviewer and common rules;
  Python loaded python-reviewer/python-patterns; Security loaded security-reviewer/
  security-review and common security rules; filesystem persistence/concurrency
  loaded database-reviewer/postgres-patterns/database-migrations. All four lanes
  fit the default three-specialist budget. Framework/ML/RAG/healthcare/network lanes
  omitted: no corresponding runtime or framework changes. Python, Security and
  persistence each concluded verbatim: "No concrete task-relevant findings."
- Fresh candidate-validation Task loaded code-reviewer/General guidance and
  validated the sole candidate T124-G1; no rejected or duplicate candidates.
  Validated finding verbatim:

  > FINDING T124-G1 — tests
  > Severity: medium
  > Evidence: `docs/release-v0.3.0.md:139-140` says the full workflow gate results and independent review dispositions are recorded in “T124 execution notes.” The tracked T-124 task file has no execution-notes section, and the only T-124 artifact found is the CLI walkthrough transcript; it does not record the full validation chain or review dispositions.
  > Finding: The candidate readiness document points readers to execution evidence that is not present in the reviewed checkout.
  > Failure/impact: A maintainer relying on the candidate record cannot verify the stated full-gate results or review dispositions from the referenced notes, despite the document presenting them as recorded.
  > Recommended direction: Add a dated execution-notes section or an actual durable evidence reference containing those results and dispositions; otherwise revise the claim to identify the evidence that is available.

  Disposition: fixed by this durable execution section. Documentation-only;
  per implementation notes, no artificial prose TDD. Direct reproduction:
  `rg -q '^## Workflow-v3 execution evidence' planning/tasks/T-124-validate-integrated-v0-3-0-workflows-offline-and.md`
  exited 1 before this addition, then 0. No deferred findings. Fresh final gate
  and disposition-verifier outcomes follow after execution, before verify/complete.
- Final recheck at 2026-10-16T15:23:08Z: exact `uv run ruff check` ->
  `uv run mypy` -> `uv run pytest -v` passed (169 source files, 2,478 tests,
  113.50s, one existing serialization warning). Then sequential `mise run check`
  passed: ruff, 383 formatted files, mypy, 2,478 tests (112.88s), commit/push/
  author/mutation guards, offline distribution inspection self-tests (54 cases)
  and publish-guard synthetic checks. Full suites never ran concurrently.
  `git diff --check` and Taskrail structural validation passed. Golden legacy
  characterization tests and claims-v1 cases passed in both full suites; historical
  diff remains empty. This is one review/disposition cycle, one documentation fix.
- Fresh final disposition-verification Task loaded code-reviewer plus General,
  Python/patterns, Security/review and persistence/database companions. Inspected
  actual diff, matching link/heading, lock hash and both saved final gate logs;
  `git diff --check` passed. Conclusion verbatim: "T124-G1 — RESOLVED."
  New findings verbatim: "No concrete task-relevant findings." No unresolved or
  deferred findings. No verify/complete commands ran before this verification.
  Captured non-Vulgate walkthrough bytes archive SHA-256:
  `ee752f9b7bea3b01cc8c003881859981645a5895d51d22f48caacb7ad6553669`;
  Vulgate full output is represented by the committed oracle/hash/byte count.
- 2026-10-16T15:25:03Z: verification pass
- 2026-10-16T15:25:03Z: Validated 2026-10-16 after all acceptance walkthrough lanes, metadata RED/GREEN, 2478-test exact chain and mise check, simplifier, four review lanes, G1 documentation fix and fresh final disposition verification; all-other-work guard passed immediately before verify. Candidate only; T038 remains blocked maintainer gate.
- Finalization guard independently re-read status/selector and asserted active
  spec v0.3.0, active count 1 owned by T124, active-spec open count 2 and blocked
  IDs exactly T038 immediately before verify; spec hash matched the pin. Verify
  pass and complete both executed at 2026-10-16T15:25:03Z, then validate passed.
  Final state: 126 completed, active 0, active-spec open 1 (only blocked T038),
  away T128 still todo/v0.8.0. Observed selector: "no active-spec eligible task",
  warning skipped_non_active_spec for T128. No second task started; outer synthetic
  adversarial acceptance remains separate (this walkthrough counts as zero rounds).
