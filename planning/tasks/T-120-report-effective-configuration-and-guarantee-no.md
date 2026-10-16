---
id: T-120-report-effective-configuration-and-guarantee-no
title: Report effective configuration and guarantee no hidden prompts across the CLI
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies:
    - T-118-apply-planned-annotation-operations-atomically
updated_at: "2026-10-16T12:58:19Z"
---

# T-120-report-effective-configuration-and-guarantee-no Report effective configuration and guarantee no hidden prompts across the CLI

## Description

Finish `specs/v0.3.0.md#agent-ready-cli-contract` with `corpus config [--policy FILE]
[--corpus FILE] [--json]` (effective configuration with value origins) and the no-prompt
guarantee: prompts anywhere in the CLI run only behind the shared `_is_interactive()`
check, `setup` also skips them under `--non-interactive`, and a needed but disallowed
prompt exits 2 with `prompt_required` naming the flags, or follows the documented default
(`cloze --parallel-mode prompt` ignores translations with a warning). Today
`_is_interactive()` lives in `cli.py` and guards only `_resolve_parallel_columns`;
`setup_flow.py` calls `typer.prompt`/`typer.confirm` (`_prompt_for_field_choices`,
`_prompt_for_corrections`, the save confirmation) unguarded. `--version` is T-103.
Other legacy commands keep their exit semantics and output.

## Acceptance

- `corpus config --policy POLICY --corpus CORPUS --json` emits one `CliResult` whose
  `result` reports tool version, supported format versions (every `latinitas-*` format and
  its maximum version), rule versions (`text_canon: 1`, `latin-fold/1`, `latin-seg/1`,
  `latin-tok/1`), resolved paths, the corpus file and header digests and the policy
  digest, each value as `{value, origin}` with origin `default`, `file` or `flag`. One
  test shows all three origins: the policy path is `flag`, `receipt_actor` is `file` when
  the policy sets it and `default` (`human`) when it does not; without `--policy` the
  policy digest is `null` with origin `default`. Unreadable or invalid inputs exit 3 with
  the loaders' codes (`unreadable`, `invalid_json`, `unsupported_format`,
  `unsupported_version`, `invalid_policy`; a corpus with findings gives one
  `invalid_corpus` error whose message names the first finding code, plus the findings);
  no file is written.
- A shared `src/latinitas_cards/interaction.py` defines `_is_interactive()` (stdin and
  stdout are TTYs) and a `prompt_required` error naming flags. Callers use
  `interaction._is_interactive()` via the module attribute (`from . import interaction`),
  so a single monkeypatch covers `cli._resolve_parallel_columns` and `setup_flow`; the
  static scan accepts both `_is_interactive()` and `interaction._is_interactive()` as
  guards.
- Static test: an AST scan of every module under `src/` finds each call to `typer.prompt`,
  `typer.confirm`, `click.prompt`, `click.confirm`, `input`, `rich.prompt.Prompt.ask` and
  `Confirm.ask` and asserts it sits inside an `if` whose test calls `_is_interactive()`,
  or in a function whose first statement is `if not _is_interactive(): raise ...`; a
  synthetic unguarded source passed to the scanner is reported (negative control). A
  second static test walks the Click tree of `app` and asserts no option sets `prompt` or
  `confirmation_prompt`.
- `setup` without `--non-interactive`, run as a subprocess with `stdin=DEVNULL` and a 20 s
  timeout: a source needing a field choice exits 2, stderr names `prompt_required`,
  `--lexical-entry-field`/`--principal-parts-field` and `--confirm`; a source needing only
  confirmation exits 2 naming `--confirm` and `--non-interactive`; `setup INPUT --profile
  P --confirm` on a source with no field choice (and with `--lexical-entry-field ...
  --principal-parts-field ... --confirm` on the field-choice source) succeeds and saves the
  profile. `--non-interactive` keeps its v0.2.x
  results (including the documented cancelled outcome).
- `cloze` on a committed parallel CSV fixture with `--parallel-mode prompt` as a
  subprocess with `stdin=DEVNULL` and a 20 s timeout exits 0, warns on stderr that
  translations are ignored, and writes no `translation_*` column.
- The existing `tests/unit/profile_setup_test.py` cases that answer prompts through
  CliRunner `input=` monkeypatch `interaction._is_interactive` to return `True` and keep
  their assertions; every new contract command (`corpus`, `annotations`, `policy` groups)
  run under CliRunner with empty stdin never blocks or reads stdin. Matrix row `corpus
  config: 0, 3`; `prompt_required` is documented in the registry.
- Pass the mandatory ruff/mypy/pytest chain; `mise run check` passes.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the static-scan call count and the subprocess exits and durations.

## Implementation Notes

- New `interaction.py` and `effective_config.py` (pure report builder); the `corpus
  config` callback is a thin addition to `commands/corpus.py` (from T-105) named
  `corpus_config`, using `run_contract`. Do not grow `cli.py`: replace its
  `_is_interactive` definition with the `interaction` module attribute call.
- In `setup_flow.py` guard `_prompt_for_field_choices`, `_prompt_for_corrections` and the
  save `typer.confirm`; when not interactive and `--non-interactive` was not given, raise
  `prompt_required` with exit 2 instead of reading stdin. `setup` stays a legacy command
  (text or its own JSON); do not add a `CliResult` envelope to it.
- Origin `file` means a value read from `--policy` or `--corpus` contents; there is no
  project file (v0.7.0). Resolved paths may be absolute here (read-only report).
- TDD order: static scan RED against current `setup_flow.py`, interaction module, setup
  guards, subprocess tests, `corpus config`. Size guard: two small modules plus guarded
  call sites.

## Execution Evidence (2026-10-16)

- Ownership/base gate: fetched origin/main; HEAD and remote both
  `897f70aee45487ae1250fa705c94ae58c5ef90a6`, clean worktree, no active task.
  Taskrail validate passed; next --json selected this todo task before start.
  Pinned specs/v0.3.0.md SHA256:
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
- Understand: source ownership is setup_flow, cli parallel-column selection,
  commands/corpus, corpus_io/corpus_validation, policy and json_schemas. The
  registry already documents prompt_required. No new schema/exporter changes.
- RED: interaction_test static scan counted 14 calls and failed on unguarded
  setup_flow (1 failed, 2 passed). GREEN after shared interaction module and
  guarded sites: scan and existing setup tests, 35 passed.
- DEVNULL subprocess RED exposed standalone Click UsageError incompatibility
  with Typer's vendored exceptions (exit 1 instead of 2); typer.BadParameter
  fixes the public exit contract without changing legacy output. Cloze's Rich
  stderr wrapping is normalized in the assertion, not changed in production.
- DEVNULL post-simplification evidence (20-second timeout per process): setup
  choice=false/confirm=false exit 2 in 0.773s; true/false exit 2 in 0.785s;
  false/true exit 0 in 0.614s; true/true exit 0 in 0.661s. Cloze exit 0 in
  0.574s, stderr warns, no translation_* column. Inputs unchanged, profiles
  exist only for successful confirmed runs. Prompt-call count remains 14.
- Config RED: 13 failures because command was absent; GREEN: 13 passed.
  A second RED (2 failures) exposed omitted legacy managed/baseline formats;
  the report now includes all twelve supported latinitas-* formats at v1.
  Origin, digest, loader-error and no-byte/mtime-change assertions pass.
- Initial broad run exposed two test integration issues: Typer's test runner
  rejects stream objects, and the existing command-registration assertion
  needed config. The no-read matrix now uses the repository's Click runner
  cast pattern with a failing-on-read empty stream. All 34 matrix cases pass.
- Exact initial chain: uv run ruff check; uv run mypy; uv run pytest -v:
  all passed, 2445 tests and one existing Pydantic serializer warning in
  116.34s. Ruff format --check: 378 files already formatted.
- Dedicated Task loaded personal code-simplifier and inspected tracked plus
  untracked scope: "No changes made." It rejected consolidation that would
  add indirection or obscure acceptance coverage. Post-simplifier focused
  interaction/config/matrix/profile tests: 105 passed in 6.16s.
- Independent parallel read-only Task lanes: General (ECC code-reviewer),
  Python (ECC python-reviewer, python-patterns and common rules), Security
  (ECC security-reviewer, security-review and common/security). All loaded
  personal code-reviewer and reviewer-index routing. Each returned verbatim:
  "No concrete task-relevant findings." Python and Security each reran the
  105-test focused suite; General checked diff whitespace.
- Fresh candidate-validation Task loaded personal code-reviewer and relevant
  reviewer references, inspected actual scope/acceptance and ran 156 tests.
  Verbatim: "No concrete task-relevant findings." Empty candidate set: no
  rejected IDs, no deduplication, no findings to fix or defer.
- Lane rationale: General always required; Python covers changed language and
  exception/test contracts; Security covers untrusted inputs and no-write/
  consent boundaries. No database, concurrency, framework, ML, RAG, healthcare
  or network changes; those lanes omitted. One review cycle, no findings.
- Scope: two focused production modules, thin config callback, guarded setup
  sites, reduced cli.py, tests, fixture and user docs. T038 stays blocked;
  T127 exporter remains unfixed/todo; T128 remains excluded/unstarted.
- 2026-10-16T12:58:19Z: verification pass
- Final tested implementation revision:
  `13c65c6120317a7fddebadb5d7eafbe3255d8d84`. Exact final chain passed:
  uv run ruff check; uv run mypy (166 files); uv run pytest -v (2445 passed,
  one existing corpus_io serializer warning, 114.45s). mise run check also
  passed: formatting (378 files), mypy, 2445 tests (111.80s), commit/push
  message, author, mutation-result and mutation-configuration guard suites.
- Scratch no-stdin negative control used the same failing-on-read stream:
  input() failed with AssertionError as intended. No scratch files retained.
- Fresh final disposition-verification Task loaded personal code-reviewer,
  routing plus General/Python/Security references, inspected the committed
  diff and acceptance, checked whitespace and reran 105 focused tests.
  Verbatim: "No concrete task-relevant findings." Empty disposition verified;
  no unresolved/new issues, no deferrals. This preceded lifecycle finalization.
- Taskrail chronology: start at 12:36:53Z, verify pass and complete at
  12:58:19Z on 2026-10-16, only after final checks and disposition verification.
  Validate passed before and after. Status: 120 done, 0 active, 1 blocked,
  7 todo; observed next T-121-complete-package-metadata-notices-and-distribution
  only, not started. Existing orphan/drift signal remains outside this task.
- Delivery preparation: hooks installed before first commit, Felix Müller
  identity and no attribution trailers. Commit-msg rejected a long body line;
  it was wrapped and all hooks passed. Final pre-push fetch still found
  origin/main at the accepted base, an ancestor of the tested implementation.
  Finalization changes only planning metadata/evidence, not tested code.
