---
id: T-147-inspect-only-active-code-in-mutation-cli-contract
title: Inspect only active code in mutation CLI contract guards
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies: []
updated_at: "2026-10-20T20:13:21Z"
---

# T-147-inspect-only-active-code-in-mutation-cli-contract Inspect only active code in mutation CLI contract guards

## Description

The generated baseline reached after T-146 stops at the CLI diagnostic-code
source guard (677 passed). It scans inactive mutant bodies, including an
intentional ContractError positional-argument deletion, as if they were active
application code. A focused generated-sandbox run also reproduces the same
problem in the prompt guard (2 failed). Both enforce the existing
agent-ready-cli-contract acceptance; neither is a product regression.

## Acceptance

- Keep the closed diagnostic-code enum and guarded-prompt source checks intact.
- Inspect original bodies and the selected mutant, never inactive variants;
  distinguish identically named functions in different modules.
- Ordinary runs still scan every function. Real original violations and active
  mutant violations must fail. Do not skip either guard under mutation testing.
- Share the minimal AST filtering between both guards, including async bodies.
- Pass focused red/green regressions, full validation, generated baseline and
  real scoped mutation smoke without claiming a full mutation campaign.

## Verification Notes

- Red: generated CLI and prompt static guards both failed (103.41 seconds).
- Red: synthetic inactive-body regression cases both failed with the expected
  false positives (literal diagnostic line 5 and unguarded prompt line 5).
- Green: generated CLI and prompt static guards both pass (215.07 seconds
  while other validation runs shared this orb). Neither guard is skipped.
- Focused scanner suite: 33 passed. Combined scanner and golden harness
  review-validation suite: 40 passed. Controls preserve original violations,
  active mutant violations, module discrimination and ordinary-source scanning.
- Dedicated personal code-simplifier Task made no edits. Independent General
  review returned "Candidate findings: none." Python returned
  "No concrete task-relevant findings." Security returned
  "No concrete, reproducible bypass or regression met the reporting threshold."
  Fresh candidate validation confirmed no concrete findings or contradictions.
  No findings required fixes or deferrals. General/Python cover guard behavior;
  Security covers preservation of the no-hidden-prompt acceptance gate.
  Database/framework lanes do not apply to this test-only AST change.
- Combined mandatory final chain passed: Ruff; mypy (170 source files);
  pytest -v (2,637 passed, 7 skipped, existing serializer warning).
  Final focused recheck passed Ruff, mypy and 40 tests; mutation shell suites,
  formatting and Taskrail validation passed.
- Real scoped command completed with exit 0:
  `uv run mutmut run --max-children 2
  latinitas_cards.claim_sidecar.x_load_sidecar__mutmut_1`.
  Full stats collection, selected clean tests and forced-failure sanity passed;
  exactly one selected mutant was killed out of 27,155 discovered. The selected
  status passes the unchanged result parser, not the full report completeness
  gate (unselected mutants have deliberately not been executed).
- Independently activated that mutant against
  `test_deck_loader_rejects_v2_before_validation`: expected failure
  `DID NOT RAISE ValueError`, proving a real behavioral kill rather than a
  harness/import failure. The ordinary full suite passes unchanged behavior.
- Fresh disposition-verification code-reviewer Task inspected both tasks and
  final evidence: "No concrete task-relevant findings." No unresolved or newly
  introduced issues; one review cycle per task, no findings to dispose.

## Implementation Notes

T-146 was temporarily blocked on this newly exposed baseline failure, then
resumed after this fix passed. Both tasks remain within existing active-spec
anchors; T-038 and T-128 are unchanged.
- 2026-10-20T20:13:21Z: verification pass
- 2026-10-20T20:13:21Z: Preserve CLI source gates while filtering only inactive mutation variants. Final independent disposition review found no concrete issues; full local chain and scoped mutation smoke passed.
