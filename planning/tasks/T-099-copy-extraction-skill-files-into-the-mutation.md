---
id: T-099-copy-extraction-skill-files-into-the-mutation
title: Copy extraction skill files into the mutation baseline sandbox
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#goals
dependencies: []
updated_at: "2026-10-13T23:31:33Z"
---

# T-099-copy-extraction-skill-files-into-the-mutation Copy extraction skill files into the mutation baseline sandbox

## Description

Scheduled GitHub Actions run 38199240879 on 2026-10-12 fails its unmutated
baseline before producing a completed mutation result:
https://github.com/fmueller/latinitas-cards/actions/runs/38199240879

The log reports FileNotFoundError for
`.agents/skills/extracting-authored-notes/SKILL.md` in
`test_extraction_skill_is_mirrored_and_documents_identity_and_safe_workflow`.
That test also reads the mirrored `.claude/skills/` file. Current
`tool.mutmut.also_copy` includes docs and planning/tasks but neither skills
directory. Source and test inspection on pinned
f8cd5dc85df946aa9a66ac5a74d6edab07cfff64 independently confirms the missing inputs.

Deduplication: completed T-039 restored then-existing docs/release/task inputs;
the later extraction skill contract adds different repository-file dependencies.
This is a separate CI harness issue, not a native appearance/product finding.

Release-scope reconciliation: the parent accepted this finding as verification
readiness required for the authorized v0.2.1 release. Its extraction contract
supports the active goals of reviewed CLI workflows and reusable exports. The
open task was explicitly repointed from the historical v0.1.0 release gate to
the discovered v0.2.1 goals anchor using Taskrail dry-run and actual repoint.
This does not authorize off-spec product work or dependency remediation; default
priority, dependencies and authoritative selector ordering remain unchanged.

## Acceptance

- Include both extraction skill contract files in the mutation sandbox.
- Verify the extraction-skill test runs against the copied sandbox inputs and
  keep the mirrored-content/identity/safe-workflow assertions intact.
- Complete an unmutated sandbox baseline and a scoped mutation smoke without
  claiming a full mutation campaign from a scoped result.

## Verification Notes

- Independently confirmed on 2026-10-13 by reading the actual failed-run log,
  current pyproject.toml also_copy configuration, and the test's exact paths.
  The remote failure is on 0be494a55aa377de397cab2371d208e19514a1d9; the omission
  remains in the ROUND5 pinned source. No workflow rerun or full mutants run.

## Implementation Notes

Do not exclude the contract test or weaken completed-baseline validation to
make the CI result appear green. No dependency alert remediation is included.
- 2026-10-13T23:31:33Z: verification pass
- 2026-10-13T23:31:33Z: Both extraction mirrors and current repository-test inputs copied. Unmutated generated sandbox: 1050 passed; scoped smoke: one mutant killed, not a full campaign. General/Python reviews and fresh validation/disposition verification found no concrete task-relevant findings. Final Ruff/mypy/pytest passed on 2026-10-13.
