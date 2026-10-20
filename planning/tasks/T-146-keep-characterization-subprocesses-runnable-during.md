---
id: T-146-keep-characterization-subprocesses-runnable-during
title: Keep characterization subprocesses runnable during mutation stats
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies: []
updated_at: "2026-10-20T20:13:21Z"
---

# T-146-keep-characterization-subprocesses-runnable-during Keep characterization subprocesses runnable during mutation stats

## Description

Restore the characterization harness required by the USFX-normalization
byte-identity acceptance contract during mutmut's unmutated stats phase.
GitHub run 39193264262 (2026-10-19) stops at the principal-parts claim golden
before executing mutants. A generated-package subprocess with
MUTANT_UNDER_TEST=stats reproduces mutmut's configuration FileNotFoundError
from the disposable working directory. This is a harness failure, not a
product regression or an efficacy-floor failure.

## Acceptance

- Keep exact CLI subprocess invocation, isolated environment, golden bytes,
  input/output audits, mutation identity and baseline completeness gates.
- Make mutmut configuration available during child execution without leaving
  harness configuration in the audited fixture directory.
- Expose child stderr in failure diagnostics and regression-test stats mode.
- Pass the generated sandbox baseline and a scoped mutation smoke; distinguish
  these from a full mutation campaign and remote scheduled verification.

## Verification Notes

- Historical runs inspected: 36372713163 missing docs; 37258605047 and
  38199240879 missing extraction mirrors (already fixed); 37116944401 one
  interrupted mutant (subsequent 37118942802 successful). Latest failure:
  https://github.com/fmueller/latinitas-cards/actions/runs/39193264262
- The October 3 successful artifact records 6,527 killed, 4,475 survived,
  221 no-tests and 2 timeouts out of 11,225. Surviving mutants are real reported
  outcomes, but do not cause the existing report-only efficacy gate to fail.
  Individual survivors were not semantically triaged in this harness task.
- Latest failure happens after dependency installation succeeds, before mutant
  execution: not an installation/environment failure. The clean ordinary suite
  passes; no product bug is established by the inspected failed runs.
- Local original mutmut stats run reproduced the same stopping point:
  1 failed, 602 passed, 7 skipped. An independent generated CLI subprocess with
  MUTANT_UNDER_TEST=stats exposed the hidden configuration FileNotFoundError.
- Strict TDD: extended subprocess regression failed twice before implementation
  (stats config unavailable; named-mutant stderr discarded), then passed both
  parameter cases, including config cleanup after success and child failure.
- Initial mandatory chain: Ruff passed; mypy passed (169 source files);
  pytest reported 2,634 passed, 7 skipped, one existing serializer warning.
- Dedicated personal code-simplifier Task recommended no edits. General,
  Python and Security code-reviewer Tasks independently returned verbatim:
  "No concrete task-relevant findings." Fresh candidate-validation Task
  confirmed the empty finding set. No findings needed fixes or deferrals.
  Security covers the disposable filesystem/subprocess boundary; Python covers
  context-manager/error behavior. No database/framework/domain lanes apply.
- Mutation result, differential/full runner and mutation-config shell suites
  passed. Formatting and Taskrail structural validation passed.
- With the separately anchored T-147 static-guard fix integrated, final Ruff,
  mypy and pytest -v passed (2,637 passed, 7 skipped); the final focused recheck
  also passed (40 tests). Four claim-golden checks passed against generated
  source with MUTANT_UNDER_TEST=stats, including the formerly failing case.
- The real scoped mutmut command in T-147 completed stats collection, selected
  clean tests and forced-failure sanity, then killed its one selected mutant.
  This is not a full 27,155-mutant campaign or a remote workflow rerun.
- Fresh disposition-verification code-reviewer Task inspected both fixes and
  final evidence: "No concrete task-relevant findings." No unresolved findings.

## Implementation Notes

No release, future-spec T-128, product output, threshold or test exclusion changes.
Config is linked only for the duration of invoke; existing fixture config is not
overwritten. Stderr is surfaced only on unexpected subprocess return codes.
The existing mutmut v3 limitation remains: stats recorded in standalone child
processes do not merge into the parent coverage map. This fix neither adds that
limitation nor claims subprocess-only golden coverage as mapped mutant coverage.
- 2026-10-20T19:54:28Z: Generated baseline now passes claim goldens but stops at CLI static guard scanning inactive mutant bodies; separate active-spec CLI-contract harness fix required.
- 2026-10-20T20:13:21Z: T-147 completed; combined generated stats and real mutation smoke now pass.
- 2026-10-20T20:13:21Z: verification pass
- 2026-10-20T20:13:21Z: Reproduced and fixed latest remote harness failure, then integrated separately anchored T-147 guard fix; full local chain and scoped mutmut passed. No full campaign or remote mutation rerun claimed.
