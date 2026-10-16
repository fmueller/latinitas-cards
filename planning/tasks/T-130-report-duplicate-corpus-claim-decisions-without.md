---
id: T-130-report-duplicate-corpus-claim-decisions-without
title: Report duplicate corpus claim decisions without internal errors
status: completed
priority: high
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies: []
updated_at: "2026-10-16T17:07:53Z"
---

# T-130-report-duplicate-corpus-claim-decisions-without Report duplicate corpus claim decisions without internal errors

## Description

Round 1 synthetic CLI testing found an internal error when a persisted version-2
claims file contains two identical decisions. All three joined readers crash
into exit 70 instead of reporting controlled input/conflict diagnostics.
Severity: medium availability/diagnostic failure on an external input; priority
high for the agent-ready contract. No acceptance bypass or mutation observed.

Contract: `specs/v0.3.0.md#agent-ready-cli-contract` and the version-2 joins in
`#sidecar-annotation-model`. Exit 70 is for unexpected internal failures; review
conflicts must be ignored/reported without corrupting accepted/open semantics.
The fix must select a diagnostic from the closed registry, or deliberately
extend its documented schema/registry, rather than leak a legacy string.

Tested revision: `8acdba8e15940e90529c9af80be3609eb3530b49`, Linux,
Python 3.14.2. Spec SHA256:
`fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.

Reproduce with the installed executable in an external temporary directory:

```python
import json, pathlib, subprocess, tempfile

with tempfile.TemporaryDirectory() as temporary:
    d = pathlib.Path(temporary)
    (d / "source.txt").write_text("Mater filium amat.\n")

    def run(*args):
        p = subprocess.run(
            ["latinitas-cards", *args, "--json"],
            cwd=d,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(p.returncode, p.stdout)
        return json.loads(p.stdout)

    run("corpus", "normalize", "source.txt", "--adapter", "text", "--corpus-id", "probe", "--output", "corpus.jsonl")
    run("annotations", "segment", "corpus.jsonl", "--output", "sidecar.jsonl", "--timestamp", "2026-01-02T03:04:05Z")
    report = run("claims", "list", "corpus.jsonl", "--annotations", "sidecar.jsonl")
    run(
        "claims",
        "record",
        "corpus.jsonl",
        "--annotations",
        "sidecar.jsonl",
        "--fingerprint",
        report["claims"][0]["fingerprint"],
        "--status",
        "accepted",
        "--reviewer",
        "Tester",
        "--reason",
        "Checked",
    )
    commands = [
        ["corpus", "show", "corpus.jsonl", "--annotations", "sidecar.jsonl"],
        ["corpus", "validate", "corpus.jsonl", "--annotations", "sidecar.jsonl"],
        ["annotations", "inspect", "sidecar.jsonl", "--corpus", "corpus.jsonl", "--check"],
    ]
    for command in commands:
        run(*command, "--claims", "corpus.jsonl.claims.json")
    path = d / "corpus.jsonl.claims.json"
    stored = json.loads(path.read_text())
    stored["decisions"].append(stored["decisions"][0])
    path.write_text(json.dumps(stored))
    for command in commands:
        run(*command, "--claims", "corpus.jsonl.claims.json")
```

Actual: single-decision controls exit 0/0/0; duplicate input returns 70/70/70,
one envelope with `internal` and `result: null`. Expected: a controlled
diagnostic identifying the conflicting input, with the duplicated decision
not treated as accepted; never exit 70. The exact controlled exit depends on
the chosen documented registry entry, not an assumed legacy `conflict` code.
All readers preserve corpus, sidecar and claims bytes and mtimes. This probe
deliberately edits external persisted input; normal `claims record` correctly
rejects conflicting write batches and is not alleged to produce this file.

## Acceptance

- All three readers handle duplicate persisted version-2 decisions without an
  internal failure or traceback, preserve read-only behavior and produce one
  valid CLI envelope with a documented diagnostic/exit.
- Conflicting decisions remain ignored, not accepted. A single valid decision
  remains visible and accepted as the positive control.
- Preserve existing stale/copied-binding/version rejection and legacy-v1
  claim output/exit exceptions; add coverage through public commands.
- Do not redesign claim identity or alter publication/future-spec gates.

## Verification Notes

- Observed 2026-10-16: 114 real CLI subprocesses, 261 checks; three failures,
  one per joined reader, all this issue. Independent six-command control/repro
  repeated by the coordinator: 0/0/0 versus 70/70/70. Every reader preserved
  file hashes, mtimes and directory membership. No timeouts.
- Deduplicated against all 128 pre-existing tasks, including completed and
  future work. T-125's completed G-1 was an invalid-corpus join failure; this
  reproduces on a valid corpus with current anchors and duplicate decisions.
  T-115's deeply nested JSON finding is distinct. T-103 owns the diagnostic
  registry but has no matching duplicate-decision finding. Not T-038/T-128.
- Investigation only; no implementation or lifecycle transition performed.

## Implementation Notes

`corpus_claims._resolve` emits legacy `Diagnostic("conflict", ...)` for the
duplicate. `join_corpus_decisions` calls `Code(finding.code)`, but the closed
enum has no `conflict` member. Its ValueError becomes the internal envelope.

### Execution audit (2026-10-16)

- Preflight: fetched origin/main; HEAD and remote both
  `57291716de0c09f0b96827ba8c04cd580ec53ace`, clean checkout. Spec SHA256
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`
  matched. Taskrail validate passed; status had 127 done, 0 active, 1 blocked,
  2 todo; deterministic next selected only this high-priority task. Started
  with `mise exec -- taskrail start` after all checks; no second task started.
- Understand: read this task, bounded v0.3.0 sidecar/agent-contract passages,
  diagnostic registry/docs, resolver, three command callers and existing public
  join/legacy tests. This fixes an external review-input diagnostic, not claim
  identity, publication or acceptance policy. Legacy resolver remains unchanged.
- Independent installed-executable reproduction from an external temporary
  directory: single valid decision show/validate/inspect-check exits 0/0/0;
  identical duplicate exits 70/70/70, code internal and null result. Repeated
  against archived base source with snapshots: every read preserved bytes,
  nanosecond mtimes and full directory membership; no locks created.
- Strict TDD: added public-command regression for identical and opposite
  duplicates across show, inspect, inspect-check and validate. Focused RED:
  `uv run pytest tests/unit/corpus_claims_test.py -k duplicate_claim_view_join -q`
  gave 8 failures (actual internal exit 70). Minimal boundary mapping to existing
  invalid_input plus show warning reporting made all 53 corpus-claims tests pass.
  Tests also prove single accepted controls and duplicate open status, exact
  diagnostics, and full directory/bytes/mtime preservation without reader locks.
- No new registry code/schema: document existing invalid_input for duplicate
  claim inputs (error class 3). Show and ordinary inspect warn at exit 0;
  validate and inspect-check error at exit 3. Show retains stale-decision silence;
  other stale/check semantics and central exit precedence remain unchanged.
  User-facing fix recorded under Unreleased; candidate version not recut.
- Post-fix external subprocess probe repeated the self-contained workflow:
  controls 0/0/0 with empty diagnostics; duplicates 0/3/3 with invalid_input,
  "Duplicate decisions ignored", claims path/fingerprint, warning/error/error.
  Every read retained bytes, mtimes, directory membership and no reader locks.
- Scratch mixed duplicate/stale probe passed: show 0 (duplicate warning only),
  validate 3 (invalid_input plus stale errors), inspect 0 (both warnings),
  inspect-check 3 (both errors), all read-only. The initial scratch input
  mismatched its two fingerprint fields and was rejected by schema validation;
  correcting both produced a valid unmatched decision and the intended check.
- Initial exact gate: `uv run ruff check` passed; `uv run mypy` passed for
  169 source files; `uv run pytest -v` passed 2496 tests with one existing
  Pydantic serializer warning (142.41 seconds).
- Dedicated simplifier Task loaded code-simplifier, inspected diff and relevant
  context, and made no edits: no demonstrable simplification preserving behavior.
  No suggestions rejected; no extra tests needed for an unchanged tree.
- Independent parallel review Tasks selected General and Python. General loaded
  code-reviewer, reviewer-index and ECC general reviewer (no companions listed).
  Python loaded code-reviewer, reviewer-index, ECC python-reviewer and
  python-patterns. Security omitted: no parser/trust/permission change; only
  existing diagnostic conversion and read-only rendering. Database omitted:
  no SQL, persistence format, locks or writer behavior changed. No framework
  lane applies to this Typer CLI; no additional domain lanes materially apply.
- General verbatim: "No concrete task-relevant findings."
  Python verbatim: "No concrete task-relevant findings."
  Fresh code-reviewer candidate-validation Task checked actual diff and both
  reports: "No concrete task-relevant findings." Zero candidates validated,
  rejected or deduplicated; no finding fixes or deferrals required.
- Final exact gate repeated sequentially: ruff passed, mypy passed (169 source
  files), pytest passed (2496 tests, one Pydantic warning, 130.92 seconds).
  Then `mise run check` passed: ruff/format (385 files), mypy, full pytest
  (2496 tests, one warning, 134.41 seconds), all commit/push/author/mutation/
  distribution/publication-guard synthetic suites. No manual workflow triggered.
- Fresh code-reviewer disposition-verification Task loaded reviewer-index,
  General/ECC common rules, Python reviewer and python-patterns; inspected the
  product commit and working audit/state. Verbatim: "No concrete task-relevant
  findings." Empty dispositions verified; no unresolved or newly introduced
  task-relevant issue. One review cycle, no finding-fix cycle necessary.
- Tested product revision: `a48b9919edc3e34d53b12141ae83dc8449abf5c4`.
  Hooks installed before first commit; author Felix Müller, no attribution
  trailers. Gate results above precede Taskrail verify/complete.
- No spec, priority, dependency, acceptance-gate, version, release, deployment,
  secret/provider/environment/infrastructure changes. T-038 remains excluded
  and blocked; T-128 remains future-spec/unstarted. Round 1 is not declared clean;
  independent adversarial round 2 belongs to the overseer, not this task cycle.
- 2026-10-16T17:07:53Z: verification pass
- After all gates and the final independent verifier, Taskrail verify passed at
  2026-10-16T17:07:53Z and complete marked only T-130 completed. Validate passed;
  status: 128 done, 0 active, 1 blocked, 1 todo. Observed next: none eligible on
  active spec; T-128 skipped as off-spec, not started; T-038 still blocked.
  Fetched before delivery: origin/main remained the accepted cumulative base
  and was an ancestor of the tested product revision; no reconciliation needed.
