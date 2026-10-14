---
id: T-120-report-effective-configuration-and-guarantee-no
title: Report effective configuration and guarantee no hidden prompts across the CLI
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies:
    - T-118-apply-planned-annotation-operations-atomically
updated_at: "2026-10-14T05:42:59Z"
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
