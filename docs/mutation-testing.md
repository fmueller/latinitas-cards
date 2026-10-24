# Mutation testing

Latinitas Cards uses mutmut to measure whether the unit suite detects changes to
Python behavior. Mutation testing supplements ruff, mypy, and pytest; it does
not replace any part of the normal validation chain.

## Current policy: establish a baseline

There is **no efficacy floor** in the initial setup. A numerical threshold from
another repository would not describe this codebase, especially while most
application logic remains concentrated in `latinitas_cards.cli`. The weekly
workflow is report-only for efficacy but still fails closed when mutation
execution fails, results are missing, or discovered mutants remain unchecked.

The first threshold decision requires a clean baseline from the default branch:

1. remove the local `mutants/` directory;
2. run `mise run test:mutate:gate`;
3. confirm the same commit completes in the `Mutation tests` workflow;
4. record raw killed, timeout, survived, and total counts per module; and
5. select a per-module floor from the observed lower distribution, explicitly
   accounting for modules with too few mutants to support a percentage.

Until that decision is recorded, low efficacy is visible but does not fail the
gate. Survivors stay in the denominator; there is no equivalent-mutant
allowlist or adjusted score. Timeouts are displayed with killed mutants as
detected behavioral changes, while their raw count remains visible. `no tests`
is also a completed, reportable outcome rather than an infrastructure failure:
it identifies mutants outside the test suite's coverage, remains in the total,
and does not fail the initial report-only gate. Unchecked, interrupted, or
unknown outcomes do fail the gate.

## Commands

```bash
# Mutate only source modules changed from main.
mise run test:mutate

# Compare with another ref.
BASE=origin/main mise run test:mutate

# Run every discovered mutant and validate report completeness.
mise run test:mutate:gate
```

Differential runs use the merge base, map changed files to exact dotted module
names, and ignore unrelated cached outcomes. They remove only
`mutants/mutmut-stats.json` before running so mutmut rebuilds its test-to-mutant
coverage map without discarding all cached mutant state.

## Automation and artifacts

`.github/workflows/mutation.yml` runs Mondays at 03:00 UTC and on manual
dispatch. It does not run for pushes or pull requests. Eight matrix jobs cover
disjoint source-file shards. Files are sorted largest first (path breaks ties)
and dealt round-robin so large modules land in different jobs. Newly added
Python files are included automatically, including package initializers.

Each disposable CI checkout adds its file list to mutmut's `only_mutate`
configuration before generation. Other modules remain importable, and every
shard still collects stats with the full test suite. Local full and differential
commands are unchanged; do not run the configuration script in a working checkout
unless you intend to restrict subsequent mutation commands to that shard.

Each job uploads `mutation-scope.json`, `mutation-results.txt`, and
`mutants/mutmut-cicd-stats.json` as `mutation-results-0` through
`mutation-results-7` for 30 days. Counts and per-module summaries describe that
shard only; sum raw counts across all eight artifacts for campaign totals.
Every shard must pass execution and result-completeness checks for the workflow
to succeed. Failures do not cancel other shards, preserving their reports.

The normal Build workflow runs only the fast shell regression tests for result
parsing, differential selection, error propagation, and configuration. It does
not execute mutants.

## Scope and interpretation

`src/latinitas_cards/` is the mutation source. Pytest collection is constrained
to `tests/` so the generated `mutants/` project cannot be collected recursively.
Mutation generation is limited to lines exercised by the test suite through
`mutate_only_covered_lines = true`. Uncovered lines are outside this report's
denominator, not evidence of effective tests. Coverage exclusions also apply;
do not add exclusions merely to improve the mutation score. No application
modules or diagnostic strings are excluded from the combined campaign.
Each CI shard allows 120 minutes; a timeout still fails the completeness gate.
Mutmut copies tests automatically; `tool.mutmut.also_copy` also includes the
documentation, release metadata, the spec index, task notes, both mirrored
extraction skill directories, and scripts used by repository-file tests
(including the native managed-update harness). Copy directories for nested
inputs, not individual files: mutmut's file-copy branch does not create parents.
Keep that list and `tests/unit/mutation_config_test.py` in sync when adding such
dependencies. Missing files abort the unmutated baseline before any mutants are
tested; they are not evidence of surviving mutations or low test efficacy.
The current command entry points are undecorated functions registered through
`app.command()(function)`, avoiding mutmut's known omission of most decorated
function bodies. A reported percentage still describes only mutants mutmut
actually discovers.

Per-module reporting is currently least informative for
`latinitas_cards.cli`, which contains most domain and command logic. As that
module is split along the repository's documented refactoring direction,
module-level results will become more diagnostic and suitable for enforcement.
