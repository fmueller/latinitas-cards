---
id: T-116-define-the-versioned-automation-policy-file-and
title: Define the versioned automation policy file and routing truth table
status: completed
priority: high
spec_ref: specs/v0.3.0.md#policy-gated-agent-automation
dependencies:
    - T-103-define-the-cli-result-envelope-exit-codes
    - T-113-define-annotation-sidecar-records-and-the
updated_at: "2026-10-16T10:43:08Z"
---

# T-116-define-the-versioned-automation-policy-file-and Define the versioned automation policy file and routing truth table

## Description

Define the `latinitas-policy` version 1 file, its routing function and `policy validate
FILE [--json]`, per `specs/v0.3.0.md#policy-gated-agent-automation` and Fixed Contracts
**Policy**, **Versioned documents** and **Identity and hashing** (`producer.kind`,
producer globs match `name`). The policy gates sidecar writes, not linguistic acceptance:
it is not part of the deck profile (`profile.py`), never enters claim fingerprints and
never routes to a claim decision. `docs/claim-review-policy.md` (automatic acceptance
disabled) stays in force. This task delivers a pure routing function and its truth table;
computing the destructive and conflicting facts belongs to T-117 and T-119, and
`annotations plan` consumes the router in T-117.

## Acceptance

- `src/latinitas_cards/policy.py` defines strict Pydantic models: `format:
  "latinitas-policy"`, `version: 1` (strict int), ordered `rules: [{operation:
  add|supersede|retract|*, layer: segmentation|morphology|translation|correction|*,
  producer_kind: rule|analyzer|agent|llm|human|*, producer: <glob>, action:
  auto|review|deny}]`, `receipt_actor: human|any` (default `human`); unknown keys at any
  level are invalid. `producer` is required (use `"*"` for any).
- `policy validate FILE --json` exits 0 for valid fixtures and 3 for each invalid one
  under `tests/fixtures/policy/`: `threshold.json`, `min-confidence.json`,
  `calibration.json` (one such field each), `unknown-field.json`, `auto-supersede.json`,
  `auto-retract.json`, `auto-any-operation.json` (`operation: "*"` with `auto`),
  `version-float.json` (`1.0`), `version-2.json`, `other-format.json`, `not-json.json` and
  a missing path; codes are `invalid_policy`, `unsupported_version`, `unsupported_format`,
  `invalid_json`, `unreadable`. A missing file reaches the callback (exit 3 with an
  envelope), not Click's `exists=True` check (exit 2).
- `route(operation, layer, producer_kind, producer_name, destructive, conflicting,
  policy)` returns `Route(action, reasons)` with `action` in `auto|review|deny` only.
  Without a policy `add` is `auto` and every other operation `review`; with a policy the
  first matching rule wins (`fnmatch.fnmatchcase` on `producer_name`), else `review`;
  destructive or conflicting operations are never `auto`: a matching `deny` still denies,
  otherwise they become `review`. Reasons are ordered tokens from a closed set documented
  in `docs/automation-policy.md` (`no_policy`, `rule:<index>`, `default_review`,
  `destructive`, `conflicting`); `rule:<index>` uses the zero-based rule index.
- Truth table: a parametrized test over operation x layer x producer kind x destructive x
  conflicting (`add` is never destructive; a `retract` can be conflicting, against a sidecar
  supersede of the same anchor and layer by the same producer: 200 combinations per policy,
  40 `add`, 80 `supersede`, 80 `retract`, asserted as `len(rows) == 3 * 200`) under no
  policy, under
  `first-match.json` (a narrower `auto` rule before a broader `review` rule, a producer
  glob such as `fixture-*`) and under `deny-precedence.json`, against expected rows in
  `tests/fixtures/policy/truth-table.json`. The expected file is produced by a generator
  in the test directory that does not import `policy.py`, and committed; at least ten
  hand-written spot rows (first match, glob miss, default `review`, `deny` over
  destructive, `deny` over conflicting) are asserted literally.
- `policy_digest(path)` is `sha256:` plus the hex digest of the file bytes, or `None`
  without `--policy`. Import tests show `policy.py` imports nothing from `claim_review`,
  `claim_sidecar` or `profile`, and `DeckProfile` gains no field.
- The policy model registered in `json_schemas.SCHEMAS`
  (`docs/schemas/latinitas-policy.v1.schema.json`, shared drift test), a new
  `docs/automation-policy.md` (rules, first match, default, never-`auto` rules, actors are
  declared not authenticated, calibrated acceptance is v0.5.0), one added paragraph in
  `docs/claim-review-policy.md` stating that the v0.3.0 policy gates sidecar writes only,
  and the matrix row `policy validate: 0, 3`.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the truth-table row count and the SHA-256 of `truth-table.json`.

## Implementation Notes

- New `policy.py` (models, loader raising contract diagnostics, `route`, `policy_digest`)
  and `commands/policy.py`, a Typer sub-app registered as `policy` in
  `_register_commands()`, using `run_contract` from T-103.
- The `auto` restriction is a model validator: any rule whose `operation` is `supersede`,
  `retract` or `*` with `action: auto` is invalid. Loader errors carry the same codes in
  `policy validate` and in `annotations plan`/`apply` (T-117, T-118).
- `receipt_actor` is parsed and exposed; its enforcement belongs to T-117, T-118 and
  T-119.
- TDD order: model rejections, validate command, router spot rows, generated truth table.
  Size guard: one module plus one command module.

## Workflow-v3 execution evidence (2026-10-16)

1. **Understand / ownership.** Fetched and verified accepted `origin/main`
   `ae37ca35c9c3ab1c103b9caf1208ea83d53b9682`; clean initial worktree, no
   product edits transferred. `taskrail validate` returned `state valid` and
   `taskrail next --json` selected exactly this task before start. Active-spec
   SHA-256 remained
   `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
   Read the full task and bounded policy, versioning and hashing contracts.
   IO Delegate doctor reported missing local provider configuration; no
   provider settings changed and subsequent direct reads were bounded.
2. **Strict TDD, in required order.** Model tests failed on missing policy
   module, then 17 passed. CLI tests failed with `No such command 'policy'`
   (exit 2); model plus CLI tests then passed (36). Literal routing tests
   failed importing missing `route`, then 15 passed (14 handwritten rows and
   the raw-byte digest test). Truth-table tests failed on the missing fixture;
   independent generation then yielded 601 passing table/completeness tests.
   Deliberately stripping bytes before hashing made the digest regression fail;
   restoring raw-byte hashing made it pass. Schema drift failed after the new
   diagnostic; generated the policy and updated envelope schema via the existing
   `render_schema` API, without changing the known exporter bug.
3. **Initial checks / real CLI.** Focused policy/schema/CLI matrix: 693 passed.
   Initial mypy exposed one test annotation issue, fixed locally. The restarted
   full chain passed Ruff and mypy (157 files), then pytest reported 2 failed,
   2243 passed: the registry documentation and command-module structure cases
   below. Real `uv run latinitas-cards policy validate ... --json` emitted
   exit-0 success with `{receipt_actor: human, rules: 2}`, exit-3
   `invalid_policy` for version 1.0, and exit-3 `unreadable` for a missing path.
   Text mode also exposed the declared `any` actor for deny-precedence.json.
4. **Simplify.** Dedicated Task loaded personal `code-simplifier`, inspected
   tracked and untracked changes, made no edits, and passed 660 focused tests
   plus Ruff. Rejected removing explicit strict-version validation and
   compressing routing/generator branches because they obscure contracts.
5. **Independent review.** Separate parallel read-only General, Security and
   Python Tasks loaded personal `code-reviewer` and their dedicated ECC guidance.
   Security loaded security-review and common security guidance; Python loaded
   python-patterns. General: "No concrete task-relevant findings."
   Security covers the policy trust boundary; Python covers strict models and
   CLI/module integration. Database and framework lanes omitted: no database,
   persistence, web framework or concurrency changes. Two specialist lanes are
   within the soft budget. A fresh candidate-validation Task loaded
   `code-reviewer`, validated all three candidates below, rejected none and
   found no duplicates.
6. **Findings and dispositions (all fixed).**
   - S1: "Reject duplicate JSON object keys when loading a policy; otherwise a
     policy that appears to deny an operation can be accepted and interpreted
     as auto by this loader." Nested action deny/auto and top-level rules
     duplication both failed new CLI tests (exit 0 instead of 3). A local
     object-pairs hook now rejects duplicates with `invalid_policy`; both
     regression cases pass, including code and path assertions. Documented it.
   - P1: "The new `invalid_policy` diagnostic is missing from the documented
     registry." Existing registry test reproduced the mismatch; added the
     exit-3 row with registered meaning, then that test passed.
   - P2: "The command-module structure test does not account for the newly
     registered `policy` group." Existing test reproduced callback-None failure;
     extended its existing group assertions/flattening for exactly `validate`,
     then it passed. No production workaround or broad test refactor.
7. **Fresh disposition verification.** One review/fix/recheck cycle. A fresh
   read-only Task loaded `code-reviewer` in disposition-verification mode,
   marked S1/P1/P2 RESOLVED, ran 62 focused checks and diff whitespace checks,
   and reported "No concrete task-relevant findings." Final mandatory chain:
   `uv run ruff check` passed; `uv run mypy` passed (157 files);
   `uv run pytest -v`: 2247 passed, 1 warning in 113.47 seconds.
   Fresh policy/schema/matrix rerun: 695 passed at 2026-10-16T10:42:04Z.
8. **Finalization.** `mise run check` passed: Ruff; format check (369 files);
   mypy (157 files); pytest (2247 passed, 1 warning, 116.91 seconds); commit,
   push, author, mutation-result and mutation-configuration shell checks.
   Revalidated state, selector and pinned hash before verify and complete.
   Verification passed at 2026-10-16T10:43:00Z; completed at
   2026-10-16T10:43:08Z. Observed next selector is T-117 (not started).
   State reports 116 done, 0 active, 1 blocked, 11 todo.
   Hooks installed with `lefthook install`;
   author and committer resolve to the maintainer, with no attribution trailers.

Truth table: 600 committed independent expected rows (200 per policy; each
40 add, 80 supersede, 80 retract). SHA-256:
`6a1028d2d3ca4cac587bbfa695642215c1c6a1309afd01e216ac0105fb0e5d5b`.
Generator imports only standard library, never policy implementation.

Scope: one 125-line policy module and one 26-line command module, plus required
registration/diagnostics/schema/docs/tests. No profile fields, claim imports or
fingerprints, automatic linguistic acceptance, plan/apply/actor enforcement,
legacy/golden/spec/priority/dependency/gate changes. T-127 remains an existing
todo exporter follow-up (not fixed); T-128 remains future v0.8.0 dense-memory
work (not started or an active gate); T-038 remains blocked. No new follow-up
was needed for this task.
- 2026-10-16T10:43:00Z: verification pass
- 2026-10-16T10:43:08Z: Completed after workflow-v3 review and fresh disposition verification; verify pass 2026-10-16T10:43:00Z. Ruff/mypy/pytest 2247 and mise run check passed; independent truth table 600 rows.
