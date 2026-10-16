---
id: T-132-ignore-stale-bindings-before-live-decision-duplicates
title: Ignore stale copied bindings before detecting duplicate live corpus decisions
status: completed
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-16T18:57:25Z"
---

# T-132-ignore-stale-bindings-before-live-decision-duplicates Ignore stale copied bindings before detecting duplicate live corpus decisions

## Description

Round 3 synthetic CLI testing found order-dependent acceptance when a persisted
v2 claims file contains one live decision and one copied-binding stale decision
with the same fingerprint. Stale-first suppresses the valid acceptance and
reports a false duplicate; live-first preserves acceptance. Severity: medium
review-state integrity/diagnostic failure; high repair priority. No mutation,
internal exit, or acceptance of conflicting live decisions was observed.

Contract: `specs/v0.3.0.md#sidecar-annotation-model` reuses stale/unmatched
decision reporting and requires accepted corpus annotations to remain accepted
in read-only views. Fixed Contracts **Claim sidecar version 2** requires the
copied binding as well as the fingerprint. `docs/reviewed-claim-cli.md` says
stale/unmatched decisions are reported and ignored. An ignored stale entry
must not poison a later live entry's duplicate detection.

Tested product: `cf62913bd15a52d8a6951c09d53dfe74f508bea1`, Linux,
Python 3.14.2. Spec SHA256:
`fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.

### Self-contained reproduction

Run with the installed `latinitas-cards` on PATH. This creates disposable
external synthetic data through supported CLI commands, then deliberately
constructs schema-valid persisted decision inputs. Normal `claims record`
is not alleged to produce these files.

```python
import copy
import hashlib
import json
import pathlib
import subprocess
import tempfile

with tempfile.TemporaryDirectory() as temporary:
    d = pathlib.Path(temporary)
    (d / "source.txt").write_text("Mater fīlium amat.\nPuer aquam portat!\n", encoding="utf-8")

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

    def snapshot():
        return {
            p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in d.iterdir()
            if p.is_file()
        }

    run("corpus", "normalize", "source.txt", "--adapter", "text", "--corpus-id", "r3", "--output", "corpus.jsonl")
    run("annotations", "segment", "corpus.jsonl", "--output", "side.jsonl", "--timestamp", "2026-01-02T03:04:05Z")
    claims = run("claims", "list", "corpus.jsonl", "--annotations", "side.jsonl")
    run(
        "claims",
        "record",
        "corpus.jsonl",
        "--annotations",
        "side.jsonl",
        "--fingerprint",
        claims["claims"][0]["fingerprint"],
        "--status",
        "accepted",
        "--reviewer",
        "Round3",
        "--reason",
        "Independent synthetic control",
    )
    live = json.loads((d / "corpus.jsonl.claims.json").read_text())["decisions"][0]
    stale = copy.deepcopy(live)
    stale["binding"]["passage_hash"] = "sha256:" + "0" * 64
    commands = [
        ["corpus", "show", "corpus.jsonl", "--annotations", "side.jsonl"],
        ["annotations", "inspect", "side.jsonl", "--corpus", "corpus.jsonl"],
        ["corpus", "validate", "corpus.jsonl", "--annotations", "side.jsonl"],
        ["annotations", "inspect", "side.jsonl", "--corpus", "corpus.jsonl", "--check"],
    ]
    for label, decisions in [
        ("single", [live]),
        ("stale-first", [stale, live]),
        ("stale-last", [live, stale]),
        ("two-live", [live, live]),
    ]:
        (d / "input.json").write_text(json.dumps({"version": 2, "decisions": decisions}) + "\n")
        before = snapshot()
        print(label)
        for command in commands:
            run(*command, "--claims", "input.json")
        assert snapshot() == before
```

Expected and actual, commands ordered as above:

| Input | Expected exits | Actual exits | Target review status |
| --- | --- | --- | --- |
| Single live | 0/0/0/0 | 0/0/0/0 | accepted |
| Stale then live | 0/0/1/1 | **0/0/3/3** | **open**, expected accepted |
| Live then stale | 0/0/1/1 | 0/0/1/1 | accepted |
| Two identical live | 0/0/3/3 | 0/0/3/3 | open |

Stale-first emits false `invalid_input: Duplicate decisions ignored`; inspect
also reports stale. Live-first emits only stale in inspect/checked readers;
show preserves its documented stale-decision silence. The two mixed arrays
contain precisely the same objects in reverse order. Even a policy rejecting
all repeated fingerprints would require consistent behavior, not this split.

## Acceptance

- Ignore stale/copied-binding mismatches before determining duplicate live
  decisions, independent of their placement in the persisted array. Preserve
  existing stale diagnostics, show silence, and checked exit 1.
- A single live accepted or withheld decision mixed with stale copies retains
  its status in either order across claims list/inspect and joined readers.
- Two or more identical or opposed live decisions remain ignored/open;
  show/ordinary inspect warn at 0 and validate/checked inspect fail with
  `invalid_input` at 3. Mixed real duplicates plus stale entries retain 3-over-1.
- Preserve bytes, nanosecond mtimes, directory membership and lock-free reader
  behavior, version gates, fingerprint identity and legacy v1 behavior.
- Add public CLI regression coverage; pass the mandatory validation chain.
  Do not redesign policy/apply safety or release/future-spec gates.

## Verification Notes

- Observed 2026-10-16. Lane-4 adversarial evidence retained 20 failed assertions
  across repeated mixed-order reads, all this one defect. Legacy claims list
  independently corroborates it. The 60-call T-130 matrix passes its original
  identical/opposed live duplicates, stale coexistence and read-only controls.
- Coordinator independently generated a new two-passage corpus and executed
  20 subprocesses. Four of 40 assertions failed exactly as above; all 16
  reader bytes/mtime snapshots were unchanged. No timeout or internal exit.
- Coordinator input SHA256s (stored bytes retained in the Round 3 bundle):
  corpus `cdb6dac9d913a3b066b6ad9375f95dc9676f1f570e9ac6af1e9b743a417ff331`;
  annotation `4fe8d7689603782ee658041d853fd33f786766e02a8078f186c1c914e236e6a9`;
  stale-first `b7bbc2ba3e61780f4623c2c5ecd2fce292720185378eede9917fc40a80a6f849`;
  stale-last `d42ce9f803cb58efd916d88cd2170306f17712eecd6082170ff296b943be6499`.
  The reproduction above uses a shorter corpus ID, so hashes differ by design.
- Dedupe searched all existing tasks, including completed T-129/T-130/T-131
  and future T-128. T-115 owns base decision resolution; T-125 owns joined
  views; neither tracked this ordering defect. T-130 repaired internal exit
  70 on live duplicates; its stale coexistence used a different fingerprint.
  This adjacent defect is not dismissed because T-130 is complete. There is
  scope overlap with its preservation criterion, but the original regression
  passes; record this standalone repair rather than silently reopening T-130.
  T-129/T-131 concern reserved locks; T-128 is unrelated future memory work.
  No dependency or follow-up to completed T-124; T-038/T-128 remain unchanged.

## Implementation Notes

Investigation correlation only, no fix made: `corpus_claims._resolve` adds
every fingerprint to `seen` after either stale rejection or live resolution.
A later live entry takes the duplicate branch; reversing the entries leaves
the resolved live decision intact. Keep the fix scoped to decision resolution
and its public observable contract.

## Execution Audit — 2026-10-16

### Task Outcome

| Status | Outcome | User/Product Impact |
| --- | --- | --- |
| Completed after all review and validation gates | Stale bindings no longer reserve live fingerprints | Legitimate accepted and withheld reviews survive either order; real live duplicates remain open |

### Base, Pin and Ownership

- Fetched `origin/main`; clean HEAD and remote both matched accepted cumulative
  base `152606f18462f37f868df0fa1b49a7e9d192f00f`, containing tested product
  `cf62913bd15a52d8a6951c09d53dfe74f508bea1`. No source-local-main transfer.
- Spec SHA256 matched
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
- `mise exec -- taskrail validate`: `state valid`. `next --json`: sole
  candidate T-132, high/todo, no dependencies, active spec, zero active tasks.
  `start` at 18:38:10Z made only T-132 active. No second task was started.
- Installed Lefthook before committing; checked Felix Müller maintainer identity.

### What Changed

| Area | Change | Contract served |
| --- | --- | --- |
| Shared corpus decision resolver | Continue after stale rejection, before live duplicate tracking | All reader consumers agree; stale entries cannot suppress a later live decision |
| Public CLI regression | One parameterized permutation matrix for six reader forms and two statuses | Ordering, real duplicate suppression, diagnostics, exit precedence and read-only behavior |
| Changelog | Fix bullet under Unreleased only | Explain the observable correction without recutting 0.3.0 |

### Public Reproduction and Read-Only Proof

Independently ran the self-contained task reproduction as subprocesses on the
accepted base before editing production, then again after the fix. Each run
normalized a fresh disposable two-passage corpus and segmented it through the CLI.
The following command order is show / ordinary inspect / validate / checked inspect:

| Input | Before | After | Target status after |
| --- | --- | --- | --- |
| Single accepted control | 0/0/0/0 | 0/0/0/0 | accepted |
| Stale then live | 0/0/3/3; false invalid_input; open | 0/0/1/1; stale only, show silent | accepted |
| Live then stale | 0/0/1/1 | 0/0/1/1 | accepted |
| Two identical live | 0/0/3/3 | 0/0/3/3 | open |

Both subprocess runs passed every per-input byte-hash/mtime snapshot check. The
kept regression performs 468 public CLI reads: accepted/withheld, legacy claims
list/checked inspect and joined readers, single control, every permutation of
stale+live, two stale+live, stale+two identical live, stale+three live with an
opposed status. Every read asserts exact directory membership, file bytes and
nanosecond mtimes after deleting the setup writer lock; no reader lock appears.
Two or more live entries stay open, including after a third live entry, and real
duplicate exit 3 dominates stale exit 1 in joined checked readers. Legacy checked
claims inspect retains its established exit 1. The original T-130 regression passes.

### Workflow Evidence

| Step | Evidence | Result |
| --- | --- | --- |
| 1 Understand | Read resolver, schema, CLI consumers, spec and reviewed-claim contract; inspected plan/apply | Shared resolver owns view decisions; plan/apply deliberately counts stale annotation-id decisions for safety and remains unchanged |
| 2 Strict TDD | `uv run pytest -q tests/unit/corpus_claims_test.py -k stale_copied --tb=short` before fix | 12 failed, 53 deselected; false duplicate diagnostic and checked 3 instead of 1 |
| 2 Green | Minimal production continue; `uv run pytest -q tests/unit/corpus_claims_test.py` | 65 passed |
| 3 Initial checks | Exact `uv run ruff check` → `uv run mypy` → `uv run pytest -v` | All passed; 169 source files; 2528 tests, one existing Pydantic serializer warning |
| 4 Simplify | Dedicated Task loaded personal code-simplifier and workflow-v3 reference | No edits recommended; minimal control flow retained; claims 65 passed |
| 5 Independent review | Separate parallel General, Python, Security Tasks, each loading personal code-reviewer and mapped references | All returned the verbatim no-findings conclusion below |
| 5 Candidate validation | Fresh read-only code-reviewer Task in candidate-validation mode | Empty candidate set confirmed; no IDs to reject or deduplicate |
| 6 Dispositions | No concrete findings | No fixes, deferrals or finding-fix TDD required |
| 7 Recheck | Exact mandatory chain rerun after review | Ruff passed; mypy 169 clean; pytest 2528 passed, one existing warning |
| 7 Full gate | Sequential `mise run check` | Exit 0; 387 formatted files, mypy 169 clean, 2528 tests; all commit/push/author/mutation/distribution/publish-guard synthetic checks passed |
| 7 Fresh final verifier | New read-only Task loaded code-reviewer in disposition-verification mode and mapped references | No unresolved or newly introduced issues; no review blocker |
| 8 Finalize | `mise exec -- taskrail verify ... --result pass` then `complete`, at 2026-10-16T18:57:25Z | State valid; T-132 completed, 130 done, zero active, T-038 blocked, T-128 todo |

Observed next after completion: `no active-spec eligible task`, empty candidates;
Taskrail warns that future-spec T-128 is skipped. Neither T-128 nor any second task
was started. `STATE.md` was changed only through Taskrail. Final spec pin unchanged.

### Review Findings and Reference Loads

All three lanes independently returned: **No concrete task-relevant findings.**
Fresh candidate validation confirmed the empty set. No finding was omitted,
paraphrased, fixed or deferred; one review cycle, no finding-fix cycle needed.
The fresh final disposition verifier also returned verbatim:
**No concrete task-relevant findings.** Its scoped `git diff --check` passed;
it inspected the audit and existing check evidence without rerunning full suites.

- General loaded reviewer-index and `ecc/agents/code-reviewer.md` (no mapped
  companions); claims 65 passed and scoped diff whitespace check passed.
- Python loaded reviewer-index, `ecc/agents/python-reviewer.md`, python-patterns,
  coding-standards and common testing/performance/code-review rules; claims 65
  and legacy goldens/review commands 33 passed; scoped whitespace check passed.
- Security loaded reviewer-index, `ecc/agents/security-reviewer.md`, security-review,
  common security/code-review rules and packaged cloud reference (not applicable);
  claims 65 passed; record and plan/apply blast radius inspected.
- Lane selection: General mandatory, Python changed language, Security persisted
  input/review integrity. Database omitted: no SQL, schema, lock or persistence
  mechanics changed. FastAPI omitted: no web framework or Pydantic schema change.
  No UI, ML, RAG, infrastructure or other specialist trigger; within lane budget.

### Scope and Limits

- Fingerprint identity, version checks, v1 goldens and writer rejection remain
  unchanged. Stale diagnostics still block recording; this fix is not a repair
  writer or automatic acceptance. No policy/apply/authentication redesign.
- T-129/T-131 lock protections untouched. T-130 remains completed, not reopened;
  no T-124 dependency. T-038 remains maintainer-blocked; future-spec T-128 unstarted.
- No release, tag, publication, deployment, manual workflow, secret/provider,
  environment, infrastructure, spec, priority, dependency or acceptance-gate changes.
- Rounds 1/2/3 remain consumed NOT CLEAN; round 4 is exclusively overseer-owned.
- 2026-10-16T18:57:25Z: verification pass
