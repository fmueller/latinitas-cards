---
id: T-133-reject-annotation-apply-aliases-to-reserved-writer
title: Reject annotation apply aliases to reserved writer locks
status: completed
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-16T20:07:57Z"
---

# T-133-reject-annotation-apply-aliases-to-reserved-writer Reject annotation apply aliases to reserved writer locks

## Description

Round 4 synthetic CLI testing at a542fd09105b551b2f73199c85b1d0c1518e57f2
found that `annotations apply` accepts aliases to reserved writer locks, contrary
to `specs/v0.3.0.md#sidecar-annotation-model`. The spec requires new writers to
reject outputs aliasing manifests or locks directly, by symlink or hard link.

An output hardlinked to the default manifest lock returns 0 and appends one
annotation while that lock is held. An output-lock symlink to the same existing
lock returns 4/busy while held, then 0 after release, instead of 3/alias.
Dangling output-lock symlinks to the default manifest lock or a symlinked
corpus's separate logical `.lock` return 0 and create the absent reserved path.

Severity: medium filesystem-contract violation; high repair priority. No held
lock inode detachment or normalizer exclusion bypass was demonstrated: atomic
replacement detaches the output hardlink only, and the reserved inode and bytes
remain intact. Do not describe this as proven concurrent data corruption.

### Standalone reproduction

Run from the repository with its installed CLI. This uses only synthetic data
in a disposable external directory and public commands; no application imports.
The reserved file's contents deliberately form a valid empty annotation stream
to distinguish alias refusal from incidental parse failure.

```python
import json
import subprocess
import tempfile
from pathlib import Path

cli = str(Path(".venv/bin/latinitas-cards").resolve())
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    def run(*args):
        process = subprocess.run(
            [cli, *map(str, args), "--json"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(process.returncode, process.stdout.strip())
        return process.returncode

    source, corpus = root / "latin.txt", root / "corpus.jsonl"
    side, patch, plan = root / "notes.jsonl", root / "patch.json", root / "plan.json"
    source.write_text("Puella rosas portat.\nSenex aquam bibit.\n")
    assert run("corpus", "normalize", source, "--adapter", "text", "--corpus-id", "repro", "--output", corpus) == 0
    passage = next(
        row for line in corpus.read_text().splitlines() if (row := json.loads(line)).get("record") == "passage"
    )
    reserved = root / "corpus.manifest.json.lock"
    reserved.unlink()
    Path(str(side) + ".lock").symlink_to(reserved.name)
    patch.write_text(
        json.dumps(
            {
                "format": "latinitas-annotation-patch",
                "version": 1,
                "operations": [
                    {
                        "operation": "add",
                        "layer": "translation",
                        "anchor": {key: passage[key] for key in ("passage_id", "passage_hash")},
                        "value": {"language": "en", "text": "The girl carries roses."},
                        "producer": {"kind": "agent", "name": "repro", "version": "1"},
                        "created": "2026-01-02T03:04:05Z",
                    }
                ],
            }
        )
    )
    assert run("annotations", "plan", patch, "--corpus", corpus, "--sidecar", side, "--output", plan) == 0
    run("annotations", "apply", plan, "--corpus", corpus, "--sidecar", side)
    print("reserved exists:", reserved.exists(), "sidecar exists:", side.exists())
```

Expected exits: 0, 0, **3/alias**; reserved and sidecar remain absent. Actual:
0, 0, **0**, appended 1; both files exist. The absent reserved file becomes an
empty lock file (0 bytes); the new sidecar contains its header and annotation.
An ordinary independent sidecar control returns 0 and appends the same operation.

## Acceptance

- Reject apply output and output-lock aliases to known reserved writer paths,
  including the default manifest lock and logical/resolved corpus locks, with
  exit 3/alias before creating or replacing those paths.
- Cover existing/absent direct, symlink, hardlink, chained and directory aliases
  where applicable; preserve held inode, bytes and real writer exclusion.
- Keep normal apply, retry/no-op, corpus/claims/sidecar locking and receipt/policy
  behavior unchanged. No discovery of arbitrary unknown custom manifests.
- Preserve the passing T-129/T-131 segmentation and T-130/T-132 claims replays.
- Run the mandatory validation chain after implementation.

## Verification Notes

- Round 4 tested exact accepted revision and pinned v0.3.0 spec SHA256
  fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704.
- The policy/locks matrix produced 20 failed assertions for this single finding;
  all 126 mandatory prior reserved-lock cases passed (750 CLI invocations,
  1,575 assertions). No timeout or internal-70 occurred.
- Independent coordinator reproduction used different Latin text and producer,
  with 12 CLI calls: hardlink and absent-output-lock cases both return 0 instead
  of 3; ordinary output succeeds. Held manifest bytes/inode remain unchanged,
  normalizer returns 4 before and after apply and 0 after lock release.
- Reproduction harness initially selected the source record by line number;
  that KeyError was a harness error, corrected to select `record == passage`.
  Original failed harness is retained in the round evidence, not hidden.
- Evidence delivery contains ROUND4-REPORT.md, a reproducible bundle, full argv,
  stdin/timeouts, outputs, input captures, hashes and failure classifications.
  No product implementation or new verification-complete transition is included.

## Implementation Notes

Source correlation: `annotation_apply.py` protects corpus, claims, default
manifest data and locks it acquires, but omits the default manifest lock and
the separate logical corpus lock. Its suffix check catches direct `.lock`
outputs but not a normally named hardlink or output-lock alias.

Dedupe searched all task files, including completed T-129 through T-132 and
future T-128. T-129/T-131 own segmentation, which passes at this revision;
T-130/T-132 own claim resolution. T-118's completed acceptance explicitly
requires apply lock-alias protection: this is an uncovered acceptance defect,
not a claim that T-118 lacked that requirement. Reopening overlap is explicit;
this standalone high-priority task records the new counterexample. T-113's
general writer requirement overlaps; T-119's policy revalidation is unaffected.
T-128 concerns future dense-markup retention, not alias protection. No task
dependency or follow-up to T-124, release gate or spec change is requested.

## Workflow-v3 execution audit

### 1. Understand and establish ownership

- Fetched origin/main before writes: HEAD and origin/main both
  `86a833f4c6f3ec8f2b32102b9c9449108703a146`, containing tested revision
  `a542fd09105b551b2f73199c85b1d0c1518e57f2`. No source-local transfer.
- Spec SHA256 matched
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
  `mise exec -- taskrail validate` returned `state valid`; next --json
  selected exactly this high-priority todo, with zero active tasks.
- Started only this task through `mise exec -- taskrail start`.
- Source-owner inspection covered apply's preflight and append, atomic_io's
  alias/locking APIs, normalizer's default manifest/corpus locks, and claims'
  exact logical lock spelling. Production changes stay in annotation_apply.py.
  The claims lock must not follow the claims DATA symlink. The corpus lock
  continues to follow corpus DATA; its separate logical spelling is reserved,
  not an additional acquired lock. No lock redesign or custom manifest search.

### 2. Public strict TDD

- Added `test_apply_reserved_writer_alias` using the public Typer CLI, real
  filesystem links and POSIX flock, independent of the round-4 harness.
- Before production edits, ran
  `uv run pytest tests/unit/annotation_apply_test.py -k reserved_writer_alias -q`:
  **14 failed, 38 passed, 4 skipped, 58 deselected**. Failures were precisely
  manifest/logical-corpus output hardlinks and output-lock symlink, hardlink,
  chain, directory, absent-symlink and absent-chain aliases. Existing output
  hardlinks returned 0; held output-lock aliases returned 4; absent aliases
  returned 0. Both held and released calls execute before result assertions.
- An initial test-harness iteration incorrectly derived the held logical lock
  with writer_lock, which resolves DATA symlinks; corrected to flock the exact
  reserved path before recording the decisive RED above. No product fix masked
  that harness correction.
- Minimal implementation adds the default manifest lock to protected inputs,
  and checks sidecar plus non-corpus acquired locks against the logical corpus
  lock. Existing checks cover resolved corpus and exact logical claims locks.
  All preflight checks run before any writer lock can be created or held.
- GREEN: `uv run pytest tests/unit/annotation_apply_test.py -q` returned
  **110 passed, 4 skipped**. Four skips are impossible derived output-lock
  direct spellings; direct output aliases remain covered.

### Public outcome matrix and controls

| Scenario | Before | After |
| --- | --- | --- |
| Existing output hardlink to manifest/logical corpus lock | 0, append | 3/alias, no write |
| Existing output-lock link to those locks, held | 4/busy | 3/alias |
| Same output-lock links released | 0, append | 3/alias |
| Absent output-lock symlink/chain to those locks | 0, creates reserved lock | 3/alias, still absent |
| Direct/symlink/chained/directory output aliases | already rejected | 3/alias |
| Resolved corpus and exact logical claims aliases | already rejected | 3/alias |
| Ordinary partial apply/receipt/retry/replan/no-op | existing controls | pass unchanged |
| Policy/actor/denial/binding and busy controls | existing controls | pass unchanged |

- Each new case snapshots all existing top-level file bytes, asserts the same
  file set and bytes after rejection, and checks reserved inode preservation
  or absence. Corpus and claims DATA are symlinked, independently exercising
  logical/resolved paths. Held and released aliases both require 3/alias.
- Additional hardlink/manifest control: real `corpus normalize` returns 4
  before and after rejected apply while the exact reserved lock is held, then
  0 after release. The existing exact logical claims-lock test remains green;
  it returns 4/busy for ordinary apply with symlinked claims DATA and creates
  no resolved-DATA claims lock. Existing per-lock exclusion/release tests pass.
- No held reserved inode detachment, exclusion bypass or concurrent data
  corruption is claimed. The defect is reserved-alias acceptance/creation.

### 3–4. Checks and dedicated simplifier

- Exact sequential chain: `uv run ruff check` → `uv run mypy` →
  `uv run pytest -v`: **All checks passed!**, **Success: no issues found in
  169 source files**, **2580 passed, 4 skipped, 1 warning** (186.66 seconds).
- `mise run check`: exit 0, including format check, same full suite, and all
  commit/push/author/mutation/distribution/publish-guard synthetic checks.
  Suite runs were sequential, not concurrent. One existing Pydantic serializer
  warning; no test failures or suppressed checks.
- Dedicated Task loaded personal `code-simplifier`, inspected task-local diff
  and reported no changes: separate logical/resolved checks preserve error
  precedence; explicit evidence matrix should not be weakened. Targeted Ruff
  format/lint and diff-whitespace checks passed. No suggestions rejected.

### 5–6. Independent lanes, candidate validation, dispositions

- Separate parallel read-only Tasks each loaded `code-reviewer` and the
  reviewer-index mapping. General loaded ECC code-reviewer and common rules
  (also Python guidance); Python independently loaded python-reviewer and
  python-patterns; Security loaded security-reviewer and security-review;
  Database/POSIX persistence loaded database-reviewer, postgres-patterns and
  database-migrations, applying only relevant filesystem locking invariants.
- Lane rationale: General mandatory; Python changed language; Security for
  user-controlled filesystem aliases; Database for persistence/concurrency.
  Three specialists meet the default budget. Framework, other-language and
  ML/RAG/healthcare/network lanes omitted because no such behavior changes.
- Every lane's verbatim conclusion: **“No concrete task-relevant findings.”**
  Reviewers read actual diff and relevant callers/helpers/tests; they did not
  duplicate suites while mise check was running. Runtime results were supplied
  rather than independently rerun by those lanes.
- Fresh read-only candidate-validation Task loaded `code-reviewer`, checked
  the empty candidate set against code/acceptance, and returned:
  **“Candidate validation: no contradiction found.”** No validated/rejected
  candidate IDs, deduplications, deferred findings or finding-fix TDD needed.
- Review cycle count: one. No additional adversarial round was launched.

### 7–8. Final gates and delivery

- Final exact Ruff → mypy → pytest chain passed again: **2580 passed,
  4 skipped, 1 warning** (180.79 seconds), followed sequentially by a second
  successful `mise run check`, including full suite and all policy guards.
- A fresh read-only Task loaded `code-reviewer` disposition-verification mode,
  inspected final diff, audit, acceptance and related contracts and returned:
  **“Disposition-verification verdict: clean.”** It found no unresolved,
  newly introduced or deferred task-relevant findings. It did not rerun suites;
  runtime evidence came from the executed final gates above.
- No taskrail verify/complete ran before these gates. Lifecycle and delivery
  results follow below; the final remote commit is reported in the thread.
- Unreleased changelog only; v0.3.0 candidate not recut. Segmentation and
  claims production modules untouched, with their tests retained in full gate.
  T-038 stays blocked; T-128 stays unstarted. No second task or follow-up
  created, no spec/policy/release/infra changes. Overseer owns the final round
  and must stop if it is not clean; this task does not authorize another loop.
- `mise exec -- taskrail verify --result pass` and `complete` succeeded at
  **2026-10-16T20:07:57Z**, followed by `validate`: **state valid**.
  Status: **131 done, 0 active, 1 blocked, 1 todo**. Observed next --json:
  **no active-spec eligible task**; T-128 is skipped as off-spec, not started.
- Pre-delivery refetch still found origin/main at the accepted base; spec pin
  remained unchanged. Hooks installed with `mise exec -- lefthook install`;
  git identity verified as Felix Müller. Delivery is a focused non-force push
  to origin/main; exact commit and post-push containment are in the callback.
- 2026-10-16T20:07:57Z: verification pass
- 2026-10-16T20:07:57Z: Completed one workflow-v3 cycle after final checks and clean independent disposition verification; no next task started.
