---
tool: mutmut
tool_version: 3.8.1
config_file: pyproject.toml
coverage_floor: 100
test_patterns:
  - "tests/**/*_test.py"
diff_budget_minutes: 10
last_full_pass: 2026-11-21
---

# Scoped research-reader mutation adapter

Scope: measure the saved research classification reader's fail-closed guards,
using two existing public journeys, not qualify the whole evidence implementation
or identify redundant semantic matrices. The maintainer approved this bounded
setup and evidence-supported pruning for T-239. No tests have been deleted.

## Tool and isolation

The dev environment pins mutmut 3.8.1. Version 3.8.0 failed statistics collection
with pytest 9.2 because it accessed removed `item._nodeid`; 3.8.1 uses `item.nodeid`.
Keep the root mutation configuration for the existing broader workflow. Run this
calibration in an ignored copy so its selection does not replace that workflow.
No hosted inference is involved; the selected journeys use synthetic local HTTP.

From the repository root, prepare a fresh disposable workspace:

```sh
mkdir -p mutation/runs
work=$(mktemp -d mutation/runs/read-XXXXXX)
cp -R src tests "$work/"
cat > "$work/pyproject.toml" <<'EOF'
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]

[tool.mutmut]
source_paths = ["src/latinitas_cards/"]
only_mutate = ["src/latinitas_cards/research_classification.py"]
mutate_only_covered_lines = true
pytest_add_cli_args_test_selection = ["tests/integration/research_classification_test.py::test_research_complete_card_and_denial[form-de]", "tests/integration/research_classification_test.py::test_research_complete_card_and_denial[verb-noun-en]"]
EOF
```

Select, baseline and run from that copy, using the root environment:

```sh
repo=$PWD
cd "$work"
PYTHONPATH="$PWD/src" "$repo/.venv/bin/python" -m pytest -q \
  'tests/integration/research_classification_test.py::test_research_complete_card_and_denial[form-de]' \
  'tests/integration/research_classification_test.py::test_research_complete_card_and_denial[verb-noun-en]'
rm -rf mutants
/usr/bin/time -p "$repo/.venv/bin/mutmut" run --max-children 2 '*research_classification.x_read*'
```

Snapshot the finished `mutants` directory with the installed mutation skill's
`snapshot` helper before another run; use its `summary`, `open` and `diff` on
snapshots. After source or test changes, return to the repository root and repeat
both blocks, creating a new workspace with current source and tests. Clearing
`mutants` discards cached scores but does not refresh copied source or tests.
Test edits alone do not invalidate cached scores. Never compare a scoped run with a whole-module run.
The diff-run command here is the same `x_read*` command, and
applies only while production changes remain inside this calibrated function.
Other production scopes require separate estimates; this calibration does not
justify a repository-wide or whole-evidence mutation campaign.

## Calibration

Initial cold calibration on Python 3.14.7, Linux, two CPUs, mutmut 3.8.1:

- Module census: 295 generated mutants; the selected reader contains 17.
- All 17 selected mutants have covering tests. `coverage_floor` records this
  selected-mutant reachability, not statement coverage, which was not measured.
- Statistics and the clean baseline passed. Measured per-test statistics were
  27.65 seconds for form/de and 49.53 seconds for verb-noun/en.
- Selected run: 560.13 seconds wall, 729.05 user + 42.19 system seconds, two
  children; 12 killed, five survived, zero timeouts. The other 278 generated
  mutants were deliberately not run, not qualified as survivors or kills.
- Two survivors bypassed rejection of a matching saved unscorable record.
  Extending only the existing form/de journey exposed mutant 13 with
  `DID NOT RAISE ValueError` (19.76 seconds); correct source passed (27.49 seconds).
- Cold post-extension run: 424.29 seconds wall, 515.54 user + 29.88 system
  seconds, two children; 16 killed, one survived, zero timeouts. Helper `diff`
  confirmed identical populations, no regressions and both tests retained.
  Mutants 13/14 now fail the unscorable guard assertion; 15/17 also fail its
  reason assertion. No new semantic gap is inferred from those diagnostic kills.
- Survivor 16 adds `XX` around the unchanged diagnostic; its inspected equivalence
  claim is indexed in `equivalents/README.md`. The 278 excluded mutants remain
  unmeasured. These scores do not establish redundancy or authorize deletion.

The calibration copy preceded the Unicode-normalization compatibility correction
in the shared record reader. These timings are not final-candidate performance
measurements. The research reader source and selected mutant population did not
change. Default timeout multiplier 15 and constant one second were retained;
no timeout was observed, so no budget was raised to conceal a hang.

## Observable

- A valid current record is returned only after current proof reconstruction.
- Changed, foreign, stale or unscorable stored evidence raises `ValueError`.
- Exact restored evidence remains readable; research completeness never grants
  production acceptance/export authority or drops unsupported denominator slots.
- Diagnostic wording is not a byte-exact contract. A reason-bearing diagnostic
  may be asserted where the existing test distinguishes unscorable rejection.

## Checks

- Baseline green: run the two-case pytest command above twice before mutation;
  both cold mutmut statistics/clean-baseline phases passed on 2026-11-21.
- Timeouts calibrated: run the cold reader command above; initial run passed
  and post-extension run passed without timeouts on 2026-11-21. Recheck after
  source or environment changes.
- Adapter structure: run the installed mutation helper's `doctor` from the root.

## Pruning disposition

No deletion batch is proposed. The two calibrated journeys cover different
families/languages, while the retained wider matrices exercise distinct language,
phrase-family, tamper, semantic, denominator and production-authority boundaries.
The scope does not establish redundancy outside these reader guards. mutmut has
no per-test kill data; any future deletion needs cold baseline/deletion snapshots,
unchanged mutant population, helper `diff`, the skill's keep-list and restored
probes before an approved permanent deletion. A passing reduced suite is not
evidence of equivalent coverage. No deletion probe was run for this task.
