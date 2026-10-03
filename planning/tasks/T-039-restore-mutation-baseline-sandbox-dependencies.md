---
id: T-039-restore-mutation-baseline-sandbox-dependencies
title: Restore mutation baseline sandbox dependencies
status: completed
priority: medium
spec_ref: specs/v0.1.0.md#revised-architecture-release-gate
dependencies: []
updated_at: "2026-10-03T10:34:15Z"
---

# T-039-restore-mutation-baseline-sandbox-dependencies Restore mutation baseline sandbox dependencies

## Description

Restore the copied repository dependencies needed by the unmutated pytest
baseline in mutmut, preserving the revised architecture release gate tests.
The September 28 scheduled run failed while reading the extraction review note
from the sandbox, before any of its 11,225 discovered mutants were checked.

## Acceptance

- Copy documentation, release metadata, and task notes read by unit tests into
  the mutation sandbox, without excluding tests or relaxing result validation.
- Add a regression check for the repository-file dependencies.
- Complete the sandbox baseline and a scoped real mutation run, and pass the
  mandatory ruff/mypy/pytest chain and mutation shell regression suites.

## Verification Notes

- Regression red: 13 missing dependencies failed; green: all 14 cases passed.
- Initial mandatory chain: ruff passed; mypy passed (53 source files);
  pytest reported 431 passed and 10 skipped.
- Real sandbox smoke: `uv run mutmut run 'latinitas_cards.html_text.*'
  --max-children 2` completed successfully. Scoped result validation reported
  63 killed, 7 survived, 0 incomplete out of 70 mutants (90.0%, report only).
- The complete repository-wide mutation campaign and remote rerun are not
  claimed by this scoped verification.
- Workflow step 4: dedicated code-simplifier reported no changes; focused
  pytest (14 cases), ruff, and diff checks passed.
- Workflow steps 5-6: independent General and Python code-reviewer lanes and
  fresh candidate validation each returned: "No concrete task-relevant
  findings." No findings required disposition. Security, database, and framework
  lanes were omitted because no application behavior or trust boundary changed.
- Workflow step 7: final full chain again passed (431 passed, 10 skipped),
  mutation result/configuration shell suites passed, and a fresh reviewer in
  disposition-verification mode found no unresolved or new task-local concerns.
- Local edits only; no commit, push, or remote workflow dispatch performed.

## Implementation Notes

- Extend `tool.mutmut.also_copy`, keeping production code and workflow failure
  handling unchanged. Copy only `planning/tasks/`, not ignored verification
  artifacts or mutable state.
- Explain the sandbox dependency contract in the mutation testing guide.
- 2026-10-03T09:50:56Z: verification pass
- 2026-10-03T10:34:15Z: verification pass
