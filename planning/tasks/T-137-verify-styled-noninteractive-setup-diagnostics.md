---
id: T-137-verify-styled-noninteractive-setup-diagnostics
title: Verify styled noninteractive setup diagnostics
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies: []
updated_at: "2026-10-17T09:48:28Z"
---

# T-137-verify-styled-noninteractive-setup-diagnostics Verify styled noninteractive setup diagnostics

## Description

Repair the DEVNULL setup diagnostic test oracle under the existing agent-ready
CLI contract. Hosted Build runs 39018309589 and 39017090393 failed the same
two raw-stderr flag assertions on Python 3.13 and 3.14 because Rich inserts
ANSI styling inside option names. Product exit and no-write behavior is correct.

## Acceptance

- Test plain and genuinely styled subprocess diagnostics explicitly, independent
  of inherited terminal/color settings; assert the styled lane emits styling.
- Compare exact semantic flag names after Click ANSI normalization, retaining
  `prompt_required`, exit 2, timeout, source immutability and profile noncreation
  assertions, and successful explicit confirmation cases.
- Do not change product behavior, disable color globally, or weaken CI support
  for Python 3.13/3.14 and Linux/macOS.
- Complete workflow-v3 review and all required local gates before lifecycle
  completion; confirm formerly failing hosted checks on the delivered revision.

## Verification Notes

- Accepted cumulative base: d49d9b205bd7f53f814271321fd571b1c5b01201.
- Understand: `tests/unit/interaction_test.py` owns the subprocess acceptance
  test; `interaction.prompt_required` supplies the correct semantic flags.
  The pinned v0.3.0 spec digest is unchanged. No product edit is needed.
- RED: expanded explicit plain/colored matrix with original raw flag assertions:
  `uv run pytest -q tests/unit/interaction_test.py -k setup_devnull` reports
  2 failed, 6 passed, 8 deselected. Only the two colored rejection cases fail
  on `--confirm`; styling assertion passes. NO_COLOR alone retains bold ANSI
  in a CI-capable terminal, so the plain lane also sets TERM=dumb.
- GREEN: Click-unstyle diagnostic assertions; the full interaction file reports
  16 passed. Exit, timeout and no-write checks remain unchanged.
- Initial exact `uv run ruff check` -> `uv run mypy` -> `uv run pytest -v`
  chain passes: Ruff clean, mypy 169 source files, pytest 2625 passed,
  7 skipped, 1 existing Pydantic warning in 347.95 seconds.
- Simplifier: dedicated Task loaded personal `code-simplifier`; no edits,
  because further consolidation would obscure separate styling/semantic checks.
- Independent read-only lanes: General loaded ECC code-reviewer; Python loaded
  ECC python-reviewer, python-patterns and common review/testing/style/workflow/
  security guidance. Each concluded verbatim: "No concrete task-relevant findings."
  General independently reran the interaction file: 16 passed.
- Lane selection: General for acceptance and test-oracle correctness; Python
  for subprocess environment and portability. Security, database, web-framework,
  ML and other specialist lanes omitted: no corresponding product boundary or
  behavior changed. Two lanes fit the specialist budget.
- Fresh candidate-validation Task loaded `code-reviewer` in candidate-validation
  mode; no candidates to validate, reject or deduplicate, and no issue invalidated
  either lane conclusion. There are no findings to fix or defer. One cycle.
- Final exact Ruff -> mypy -> pytest chain passes: clean Ruff, mypy 169 files,
  2625 passed, 7 skipped, 1 existing warning in 345.70 seconds. Sequential
  `mise run check` passes including formatting, the full suite (2625 passed,
  7 skipped in 350.64 seconds) and every policy/infrastructure guard.
- Isolated Python 3.13 interaction file: 16 passed in 9.56 seconds after the
  default Python 3.14 gates. No test suites ran concurrently.
- Fresh disposition-verification Task loaded `code-reviewer` in that mode;
  verdict verbatim: "No concrete task-relevant code findings." No unresolved
  dispositions. It correctly identified hosted exact-head verification as the
  outstanding delivery gate, not a code finding. Implementation completion is
  local; this thread must still observe automatic hosted CI after pushing.

## Implementation Notes

- Sole write ownership released by coordinator after T136 acceptance. New task
  created and started through Taskrail; hooks installed before committing.
- Scope is the existing test file and this tracked-work audit record only.
- 2026-10-17T09:48:28Z: verification pass
- 2026-10-17T09:48:28Z: Implementation verified after workflow-v3 review and all local gates; hosted delivery verification remains with owning CI repair thread.
